<script setup lang="ts">
import { Niivue } from '@niivue/niivue'
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'

type RiskResponse = {
  ok: boolean
  risk_probability: number
  cta_path?: string
  reference_path?: string | null
  features?: {
    metadata?: Record<string, unknown>
    morphology?: Record<string, unknown>
    radiomics?: Record<string, unknown>
  }
}

const apiBase = ''
const mainCanvas = ref<HTMLCanvasElement | null>(null)
const compareCanvas = ref<HTMLCanvasElement | null>(null)

const mainViewer = new Niivue()
const compareViewer = new Niivue()

const sourceFile = ref<File | null>(null)
const compareFile = ref<File | null>(null)

const uploadedCtaUrl = ref('')
const segmentedCtaUrl = ref('')
const segmentedMaskUrl = ref('')
const compareViewerUrl = ref('')

const busy = reactive({
  upload: false,
  segment: false,
  predict: false,
})

const clinicForm = reactive({
  age: '',
  sex: '',
  hypertension: '',
  heartDisease: '',
  diabetes: '',
  atherosclerosis: '',
  drinking: '',
  smoking: '',
  hemorrhageHistory: '',
})

const statusMessage = ref('请选择 CTA 文件开始。')
const riskResult = ref<RiskResponse | null>(null)

const selectedFeatureSummary = computed(() => {
  if (!riskResult.value?.features) return []
  const metadata = riskResult.value.features.metadata ?? {}
  const morphology = riskResult.value.features.morphology ?? {}
  return [
    ...Object.entries(metadata).slice(0, 4),
    ...Object.entries(morphology).slice(0, 6),
  ]
})

async function uploadFileToBackend(file: File) {
  const formData = new FormData()
  formData.append('cta_file', file)
  const response = await fetch(`${apiBase}/api/cta/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new Error(await response.text())
  }
  return response.json()
}

function buildVolumeList(baseUrl?: string, overlayUrl?: string) {
  const volumes = []
  if (baseUrl) {
    volumes.push({
      url: baseUrl,
      name: 'CTA',
      colorMap: 'gray',
      opacity: 1,
    })
  }
  if (overlayUrl) {
    volumes.push({
      url: overlayUrl,
      name: 'Mask',
      colorMap: 'red',
      opacity: 0.55,
    })
  }
  console.log('[viewer] buildVolumeList', volumes)
  return volumes
}

async function renderMainViewer() {
  console.log('[viewer] renderMainViewer')

  const volumes = buildVolumeList(segmentedCtaUrl.value || uploadedCtaUrl.value, segmentedMaskUrl.value || undefined)
  if (!volumes.length) {
    mainViewer.closeDrawing()
    return
  }
  console.log('[viewer] renderMainViewer volumes', volumes)
  await mainViewer.loadVolumes(volumes as never[])
}

async function renderCompareViewer() {
  if (!compareViewerUrl.value) {
    compareViewer.closeDrawing()
    return
  }
  await compareViewer.loadVolumes([
    {
      url: compareViewerUrl.value,
      name: 'Comparison',
      colorMap: 'gray',
      opacity: 1,
    },
  ] as never[])
}

async function uploadSourceCta() {
  if (!sourceFile.value) {
    statusMessage.value = '请先选择 CTA 文件。'
    return
  }

  busy.upload = true
  statusMessage.value = '正在上传 CTA 文件...'
  console.log('[viewer] uploadSourceCta')
  try {
    const data = await uploadFileToBackend(sourceFile.value)
    console.log('[viewer] uploadSourceCta response', data)
    uploadedCtaUrl.value = data.file_path
    segmentedCtaUrl.value = ''
    segmentedMaskUrl.value = ''
    riskResult.value = null

    await renderMainViewer()
    console.log('[viewer] uploadSourceCta rendered')
    statusMessage.value = 'CTA 文件上传成功，已显示在左侧视图。'
  } catch (error) {
    statusMessage.value = `CTA 上传失败：${String(error)}`
  } finally {
    busy.upload = false
  }
}

async function runSegmentation() {
  if (!sourceFile.value) {
    statusMessage.value = '请先选择 CTA 文件。'
    return
  }

  busy.segment = true
  statusMessage.value = '正在执行动脉瘤分割...'
  try {
    const formData = new FormData()
    formData.append('cta_file', sourceFile.value)
    const response = await fetch(`${apiBase}/api/cta/segment`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) {
      throw new Error(await response.text())
    }
    const data = await response.json()
    segmentedCtaUrl.value = data.cta_path
    segmentedMaskUrl.value = data.mask_path
    if (!uploadedCtaUrl.value) {
      uploadedCtaUrl.value = data.cta_path
    }
    await renderMainViewer()
    statusMessage.value = data.message || '分割完成，已显示返回的 CTA 与 mask。'
  } catch (error) {
    statusMessage.value = `分割失败：${String(error)}`
  } finally {
    busy.segment = false
  }
}

async function uploadComparisonFile() {
  if (!compareFile.value) {
    statusMessage.value = '请先选择对比文件。'
    return
  }

  busy.upload = true
  statusMessage.value = '正在上传对比文件...'
  try {
    const data = await uploadFileToBackend(compareFile.value)
    compareViewerUrl.value = data.file_path
    await renderCompareViewer()
    statusMessage.value = '对比文件已上传，并显示在右侧视图。'
  } catch (error) {
    statusMessage.value = `对比文件上传失败：${String(error)}`
  } finally {
    busy.upload = false
  }
}

async function predictRisk() {
  if (!sourceFile.value) {
    statusMessage.value = '请先选择 CTA 文件。'
    return
  }

  busy.predict = true
  statusMessage.value = '正在进行风险预测...'
  try {
    const formData = new FormData()
    formData.append('cta_file', sourceFile.value)
    if (compareFile.value) {
      formData.append('reference_file', compareFile.value)
    }
    formData.append('age', clinicForm.age)
    formData.append('sex', clinicForm.sex)
    formData.append(
      'extra_metadata',
      JSON.stringify({
        高血压: clinicForm.hypertension,
        心脏病: clinicForm.heartDisease,
        糖尿病: clinicForm.diabetes,
        脑血管硬化: clinicForm.atherosclerosis,
        饮酒: clinicForm.drinking,
        抽烟: clinicForm.smoking,
        出血史: clinicForm.hemorrhageHistory,
      }),
    )

    const response = await fetch(`${apiBase}/api/risk/predict`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) {
      throw new Error(await response.text())
    }
    riskResult.value = (await response.json()) as RiskResponse
    statusMessage.value = '风险预测完成。'
  } catch (error) {
    statusMessage.value = `风险预测失败：${String(error)}`
  } finally {
    busy.predict = false
  }
}

function onSourceFileChange(event: Event) {
  const target = event.target as HTMLInputElement
  sourceFile.value = target.files?.[0] ?? null
  if (sourceFile.value) {
    uploadedCtaUrl.value = ''
    segmentedCtaUrl.value = ''
    segmentedMaskUrl.value = ''
    riskResult.value = null
    statusMessage.value = `已选择 CTA 文件：${sourceFile.value.name}，请点击“上传并显示 CTA”。`
  }
}

function onCompareFileChange(event: Event) {
  const target = event.target as HTMLInputElement
  compareFile.value = target.files?.[0] ?? null
  if (compareFile.value) {
    statusMessage.value = `已选择对比文件：${compareFile.value.name}`
  }
}

onMounted(async () => {
  if (mainCanvas.value) {
    await mainViewer.attachToCanvas(mainCanvas.value)
    mainViewer.setSliceType(mainViewer.sliceTypeMultiplanar)
  }
  if (compareCanvas.value) {
    await compareViewer.attachToCanvas(compareCanvas.value)
    compareViewer.setSliceType(compareViewer.sliceTypeMultiplanar)
  }
})

onBeforeUnmount(() => {
})
</script>

<template>
  <div class="page-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Gradulate UI</p>
        <h1>CTA 分割与风险预测</h1>
        <p class="subtext">前后端分离的 Vue + Vite 页面，支持 CTA 上传、分割结果显示、风险预测与对比查看。</p>
      </div>
      <div class="status-card">
        <span class="status-label">当前状态</span>
        <strong>{{ statusMessage }}</strong>
      </div>
    </header>

    <main class="content-grid">
      <section class="panel controls-panel">
        <h2>CTA 操作</h2>
        <label class="field">
          <span>CTA 源文件</span>
          <input type="file" accept=".nii,.nii.gz" @change="onSourceFileChange" />
        </label>
        <div class="button-row">
          <button :disabled="busy.upload || !sourceFile" @click="uploadSourceCta">
            {{ busy.upload ? '上传中...' : '上传并显示 CTA' }}
          </button>
          <button class="primary" :disabled="busy.segment || !sourceFile" @click="runSegmentation">
            {{ busy.segment ? '分割中...' : '分割动脉瘤' }}
          </button>
        </div>

        <h2>对比文件</h2>
        <label class="field">
          <span>对比 CTA / Mask</span>
          <input type="file" accept=".nii,.nii.gz" @change="onCompareFileChange" />
        </label>
        <button :disabled="!compareFile" @click="uploadComparisonFile">加载到右侧对比视图</button>

        <h2>风险预测输入</h2>
        <div class="form-grid">
          <label class="field">
            <span>年龄</span>
            <input v-model="clinicForm.age" type="number" placeholder="例如 63" />
          </label>
          <label class="field">
            <span>性别</span>
            <select v-model="clinicForm.sex">
              <option value="">请选择</option>
              <option value="男">男</option>
              <option value="女">女</option>
            </select>
          </label>
          <label class="field">
            <span>高血压</span>
            <input v-model="clinicForm.hypertension" placeholder="有 / 无" />
          </label>
          <label class="field">
            <span>心脏病</span>
            <input v-model="clinicForm.heartDisease" placeholder="有 / 无" />
          </label>
          <label class="field">
            <span>糖尿病</span>
            <input v-model="clinicForm.diabetes" placeholder="有 / 无" />
          </label>
          <label class="field">
            <span>脑血管硬化</span>
            <input v-model="clinicForm.atherosclerosis" placeholder="有 / 无" />
          </label>
          <label class="field">
            <span>饮酒</span>
            <input v-model="clinicForm.drinking" placeholder="有 / 无" />
          </label>
          <label class="field">
            <span>抽烟</span>
            <input v-model="clinicForm.smoking" placeholder="有 / 无" />
          </label>
          <label class="field">
            <span>出血史</span>
            <input v-model="clinicForm.hemorrhageHistory" placeholder="有 / 无" />
          </label>
        </div>
        <button class="primary" :disabled="busy.predict || !sourceFile" @click="predictRisk">
          {{ busy.predict ? '预测中...' : '进行风险预测' }}
        </button>

        <div v-if="riskResult" class="risk-card">
          <div class="risk-score">
            <span>风险概率</span>
            <strong>{{ (riskResult.risk_probability * 100).toFixed(2) }}%</strong>
          </div>
          <ul class="feature-list">
            <li v-for="[key, value] in selectedFeatureSummary" :key="key">
              <span>{{ key }}</span>
              <code>{{ value }}</code>
            </li>
          </ul>
        </div>
      </section>
      <section class="panel display-panel">
        <section class="panel viewer-panel">
          <div class="viewer-header">
            <div>
              <h2>主视图</h2>
              <p>显示 CTA 原图，以及分割返回的 mask 叠加结果。</p>
            </div>
            <div class="viewer-meta">
              <span>上方：CTA / Mask</span>
            </div>
          </div>
          <canvas ref="mainCanvas" class="viewer-canvas"></canvas>
        </section>
        <section class="panel viewer-panel">
          <div class="viewer-header">
            <div>
              <h2>对比视图</h2>
              <p>显示第二个 CTA 或参考 mask，便于和分割结果进行可视化对比。</p>
            </div>
            <div class="viewer-meta">
              <span>下方：对比文件</span>
            </div>
          </div>
          <canvas ref="compareCanvas" class="viewer-canvas"></canvas>
        </section>

      </section>
      </main>
  </div>
</template>
