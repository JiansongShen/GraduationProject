<script>
import { Niivue } from '@niivue/niivue'

// Keep nv outside or as a non-reactive constant to avoid Vue proxy issues with WebGL contexts
const nv = new Niivue()

export default {
  name: 'App',
  data() {
    return {
      // Start with an empty list
      volumeList: []
    }
  },
  methods: {
    async upload_files() {
      const fileInput = document.getElementById('file-input')
      if (!fileInput.files.length) return

      const file = fileInput.files[0]
      const url = URL.createObjectURL(file)

      // 1. Create the volume object
      const newVolume = {
        url: url,
        name: file.name,
      }

      console.log(newVolume)
      console.log("updated")

      // 2. Add to our local list
      this.volumeList.push(newVolume)
      console.log(this.volumeList)

      // 3. Tell Niivue to load the volumes
      // loadVolumes accepts an array
      if (this.volumeList.length > 1) {
        nv.addVolume(this.volumeList[this.volumeList.length - 1])
        console.log("updated: add new volume")
      }
      await nv.loadVolumes([newVolume])
    }
  },
  mounted() {
    nv.attachTo('gl')
  }
}
</script>

<template>
  <div class="container">
    <input type="file" id="file-input" @change="upload_files" />

    <hr />

    <canvas id="gl" height="480" width="640"></canvas>
  </div>
</template>

<style scoped>
canvas {
  background-color: black;
  display: block;
  margin-top: 10px;
}
</style>