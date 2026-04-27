<script setup>
import { nextTick, onMounted, ref } from "vue";
import { Niivue } from "niivue";

const apiBase = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

const ctaFile = ref(null);
const refFile = ref(null);
const age = ref("");
const sex = ref("");
const extra = ref("{}");
const result = ref("等待操作...");
const configInfo = ref("加载配置中...");

const viewerRef = ref(null);
let viewer = null;
let currentVolumes = [];

function getApiUrl(path) {
  if (!apiBase) {
    return path;
  }
  return `${apiBase}${path}`;
}

function resolveFileUrl(path) {
  if (!apiBase) {
    return path;
  }
  return `${apiBase}${path}`;
}

async function loadConfig() {
  const response = await fetch(getApiUrl("/api/config"));
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const data = await response.json();
  configInfo.value = `配置: ${data.config_path}`;
}

async function setVolumeFromUrl(url, name = "CTA") {
  currentVolumes = [{ url: resolveFileUrl(url), name }];
  await viewer.loadVolumes(currentVolumes);
}

function getCtaInputFile() {
  return ctaFile.value?.files?.[0] || null;
}

function getReferenceInputFile() {
  return refFile.value?.files?.[0] || null;
}

async function uploadCtaAndDisplay() {
  try {
    const file = getCtaInputFile();
    if (!file) {
      throw new Error("请先选择 CTA 文件");
    }
    const fd = new FormData();
    fd.append("cta_file", file);
    const response = await fetch(getApiUrl("/api/cta/upload"), { method: "POST", body: fd });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = await response.json();
    await setVolumeFromUrl(data.file_path, "CTA");
    result.value = "CTA 已显示";
  } catch (error) {
    result.value = error?.message || String(error);
  }
}

async function segmentCta() {
  try {
    const file = getCtaInputFile();
    if (!file) {
      throw new Error("请先选择 CTA 文件");
    }
    const fd = new FormData();
    fd.append("cta_file", file);

    const response = await fetch(getApiUrl("/api/cta/segment"), { method: "POST", body: fd });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = await response.json();
    const ctaPath = resolveFileUrl(data.cta_path);
    const maskPath = resolveFileUrl(data.mask_path);
    currentVolumes = [
      { url: ctaPath, name: "CTA" },
      { url: maskPath, name: "Mask", opacity: 0.35 },
    ];
    await viewer.loadVolumes(currentVolumes);
    result.value = data.message;
  } catch (error) {
    result.value = error?.message || String(error);
  }
}

async function predictRisk() {
  try {
    const file = getCtaInputFile();
    if (!file) {
      throw new Error("请先选择 CTA 文件");
    }
    const reference = getReferenceInputFile();
    const fd = new FormData();
    fd.append("cta_file", file);
    if (reference) {
      fd.append("reference_file", reference);
    }
    fd.append("age", age.value);
    fd.append("sex", sex.value);
    fd.append("extra_metadata", extra.value || "{}");

    const response = await fetch(getApiUrl("/api/risk/predict"), { method: "POST", body: fd });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = await response.json();
    result.value = JSON.stringify(data, null, 2);
  } catch (error) {
    result.value = error?.message || String(error);
  }
}

async function reloadView() {
  if (currentVolumes.length > 0) {
    await viewer.loadVolumes(currentVolumes);
  }
}

onMounted(async () => {
  await nextTick();
  viewer = new Niivue({ show3Dcrosshair: true });
  viewer.attachToCanvas(viewerRef.value);
  try {
    await loadConfig();
  } catch (error) {
    result.value = error?.message || String(error);
  }
});
</script>

<template>
  <div class="page">
    <header>
      <div>
        <div class="title">CTA 上传 / 分割 / 风险预测</div>
        <div class="hint">使用 NiiVue 单视窗显示 CTA 的三方向切面，支持分割参考文件与风险预测特征转化。</div>
      </div>
      <div class="hint">{{ configInfo }}</div>
    </header>
    <main>
      <section class="card stack">
        <div>
          <label>CTA 文件</label>
          <input ref="ctaFile" type="file" accept=".nii,.nii.gz" />
        </div>
        <div>
          <label>参考分割文件（可选）</label>
          <input ref="refFile" type="file" accept=".nii,.nii.gz" />
        </div>
        <div class="row">
          <button @click="uploadCtaAndDisplay">上传并显示</button>
          <button class="secondary" @click="segmentCta">上传并分割</button>
        </div>
        <div class="row">
          <button @click="predictRisk">风险预测</button>
          <button class="secondary" @click="reloadView">重新加载视图</button>
        </div>
        <div class="row">
          <div>
            <label>年龄</label>
            <input v-model="age" placeholder="如 62" />
          </div>
          <div>
            <label>性别</label>
            <select v-model="sex">
              <option value="">未知</option>
              <option value="男">男</option>
              <option value="女">女</option>
            </select>
          </div>
        </div>
        <div>
          <label>额外手工补充信息（JSON）</label>
          <textarea v-model="extra" rows="6" />
        </div>
        <div>
          <label>结果</label>
          <div class="result">{{ result }}</div>
        </div>
      </section>
      <section class="card">
        <canvas ref="viewerRef" id="viewer"></canvas>
      </section>
    </main>
  </div>
</template>
