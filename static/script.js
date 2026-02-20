// ===============================
// DOM ELEMENTS
// ===============================

let landingPage, aplikasiView;
let appSelection, appWorkspace, appTitle;
let fileInput, uploadZone, previewContainer, imagePreview;
let videoWrapper, videoCamera, canvas;
let btnOpenCamera, btnCapture, btnRealtime, btnStopRealtime, btnReset;
let resultBox, initialStateBox, resultValue, additionalInfo;

let currentMode = "klasifikasi";
let stream = null;
let realtimeInterval = null;
let isRealtime = false;

// ===============================
// NAVIGATION FUNCTIONS
// ===============================

function navigateTo(section) {
  const links = document.querySelectorAll('nav a');
  links.forEach(link => link.classList.remove('active'));

  if (section === 'aplikasi') {
    landingPage.classList.add('hidden-view');
    aplikasiView.classList.remove('hidden-view');
    document.getElementById('link-aplikasi').classList.add('active');
  } else {
    aplikasiView.classList.add('hidden-view');
    landingPage.classList.remove('hidden-view');

    const targetLink = document.getElementById(`link-${section}`);
    if (targetLink) targetLink.classList.add('active');

    const targetSection = document.getElementById(section);
    if (targetSection) {
      targetSection.scrollIntoView({ behavior: 'smooth' });
    }
  }
}

function openApp(mode) {
  currentMode = mode;
  appSelection.classList.add('hidden-view');
  appWorkspace.classList.remove('hidden-view');

  if (mode === 'klasifikasi') {
    appTitle.innerText = 'Klasifikasi Kematangan';
  } else {
    appTitle.innerText = 'Deteksi Objek Pisang';
  }

  resetApp();
}

function backToSelection() {
  resetApp();
  appWorkspace.classList.add('hidden-view');
  appSelection.classList.remove('hidden-view');
}

// ===============================
// FILE UPLOAD FUNCTIONS
// ===============================

function triggerUpload() {
  if (fileInput) {
    fileInput.click();
  }
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    showWorkspacePreview(true);
    calculateResult(imagePreview);
  };
  reader.readAsDataURL(file);
}

function showWorkspacePreview(show) {
  if (show) {
    if (uploadZone) uploadZone.style.display = 'none';
    if (previewContainer) previewContainer.style.display = 'block';
    if (imagePreview) imagePreview.style.display = 'block';

    if (videoWrapper) videoWrapper.style.display = 'none';
    if (canvas) canvas.style.display = 'none';

    if (btnOpenCamera) btnOpenCamera.style.display = 'flex';
    if (btnReset) btnReset.style.display = 'flex';
    if (btnCapture) btnCapture.style.display = 'none';
    if (btnRealtime) btnRealtime.style.display = 'none';
    if (btnStopRealtime) btnStopRealtime.style.display = 'none';
  } else {
    if (uploadZone) uploadZone.style.display = 'block';
    if (previewContainer) previewContainer.style.display = 'none';
    if (imagePreview) imagePreview.style.display = 'none';

    if (videoWrapper) videoWrapper.style.display = 'none';
    if (canvas) canvas.style.display = 'none';

    if (btnOpenCamera) btnOpenCamera.style.display = 'inline-block';
    if (btnReset) btnReset.style.display = 'none';
    if (btnCapture) btnCapture.style.display = 'none';
    if (btnRealtime) btnRealtime.style.display = 'none';
    if (btnStopRealtime) btnStopRealtime.style.display = 'none';

    resetResults();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  landingPage = document.getElementById("landing-page");
  aplikasiView = document.getElementById("aplikasi-view");

  appSelection = document.getElementById("app-selection");
  appWorkspace = document.getElementById("app-workspace");
  appTitle = document.getElementById("app-title");

  fileInput = document.getElementById("fileInput");
  uploadZone = document.getElementById("uploadZone");
  previewContainer = document.getElementById("previewContainer");
  imagePreview = document.getElementById("imagePreview");
  videoWrapper = document.getElementById("videoWrapper");
  videoCamera = document.getElementById("videoCamera");
  canvas = document.getElementById("canvas");

  btnOpenCamera = document.getElementById("btn-open-camera");
  btnCapture = document.getElementById("btn-capture");
  btnRealtime = document.getElementById("btn-realtime");
  btnStopRealtime = document.getElementById("btn-stop-realtime");
  btnReset = document.getElementById("btn-reset");
  resultBox = document.getElementById("resultBox");
  initialStateBox = document.getElementById("initialStateBox");
  resultValue = document.getElementById("resultValue");
  additionalInfo = document.getElementById("additionalInfo");

  if (btnOpenCamera) {
    btnOpenCamera.onclick = (e) => {
      e.preventDefault(); // Prevent accidental form submit if inside form
      console.log("Tombol Kamera diklik via Listener");
      startCamera();
    };
    console.log("Kamera button listener attached");
  }

  // Check if new elements exist
  if (!videoWrapper || !btnRealtime) {
    console.warn("Elements missing. Clear cache/Hard Refresh recommended.");
  }
});

// ===============================
// ANIMATION OBSERVER
// ===============================

const observerOptions = { threshold: 0.1 };
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
    }
  });
}, observerOptions);

document.querySelectorAll(".animate").forEach((el) => observer.observe(el));

// ===============================
// RESET FUNCTIONS
// ===============================

function resetApp() {
  stopCamera();
  showWorkspacePreview(false);
  fileInput.value = "";

  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.display = "none";
    canvas.style.position = 'static';
    canvas.style.transform = 'none';
  }

  if (btnOpenCamera) btnOpenCamera.style.display = "inline-block";
}

function resetResults() {
  if (resultBox) resultBox.style.display = "none";
  if (initialStateBox) initialStateBox.style.display = "block";
}

// ===============================
// CAMERA SECTION
// ===============================

async function startCamera() {
  console.log("Fungsi startCamera dipanggil");

  // Re-query elements just in case
  if (!videoWrapper) videoWrapper = document.getElementById("videoWrapper");
  if (!videoCamera) videoCamera = document.getElementById("videoCamera");
  if (!canvas) canvas = document.getElementById("canvas");

  if (!videoWrapper) {
    alert("Error Sistem: Video Wrapper tidak ditemukan. Mohon refresh halaman.");
    return;
  }

  // Check if getUserMedia is supported
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Browser Anda tidak mendukung akses kamera atau koneksi tidak aman (HTTPS required).");
    return;
  }

  try {
    // Start Camera without strict facingMode for better compatibility
    stream = await navigator.mediaDevices.getUserMedia({
      video: true
    });

    videoCamera.srcObject = stream;

    if (uploadZone) uploadZone.style.display = "none";
    if (previewContainer) previewContainer.style.display = "block";

    videoWrapper.style.display = "block";
    videoCamera.style.display = "block";

    videoCamera.onloadedmetadata = () => {
      canvas.width = videoCamera.videoWidth;
      canvas.height = videoCamera.videoHeight;
    };

    if (canvas) canvas.style.display = "none";
    if (imagePreview) imagePreview.style.display = "none";

    if (btnCapture) btnCapture.style.display = "flex";
    if (btnRealtime) btnRealtime.style.display = "flex";
    if (btnStopRealtime) btnStopRealtime.style.display = "none";
    if (btnReset) btnReset.style.display = "flex";
    if (btnOpenCamera) btnOpenCamera.style.display = "none";

    resetResults();
  } catch (err) {
    console.error(err);
    alert("Gagal akses kamera: " + err.message);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  if (realtimeInterval) {
    clearInterval(realtimeInterval);
    realtimeInterval = null;
  }
  isRealtime = false;

  // Re-query if needed
  if (!videoWrapper) videoWrapper = document.getElementById("videoWrapper");

  if (videoWrapper) videoWrapper.style.display = "none";
  if (videoCamera) videoCamera.style.display = "none";

  if (btnCapture) btnCapture.style.display = "none";
  if (btnRealtime) btnRealtime.style.display = "none";
  if (btnStopRealtime) btnStopRealtime.style.display = "none";
  if (btnOpenCamera) btnOpenCamera.style.display = "flex";
}

// ===============================
// LABEL MAPPING
// ===============================

function getMaturityLabel(class_id) {
  const mapping = {
    1: "Pisang Mentah",
    2: "Pisang Matang",
    3: "Pisang Terlalu Matang",
  };
  return mapping[class_id] || "Unknown";
}

// ===============================
// REALTIME DETECTION
// ===============================

function startRealtime() {
  if (!stream) {
    startCamera().then(() => startRealtime());
    return;
  }

  isRealtime = true;

  // UI Updates
  btnCapture.style.display = "none";
  btnRealtime.style.display = "none";
  btnStopRealtime.style.display = "flex";

  // Setup Overlay Canvas
  canvas.style.display = "block";
  canvas.style.position = "absolute";
  canvas.style.top = "0";
  canvas.style.left = "0";
  canvas.style.width = "100%";
  canvas.style.height = "100%";

  // Match canvas size to video resolution
  canvas.width = videoCamera.videoWidth;
  canvas.height = videoCamera.videoHeight;

  // Start Loop
  realtimeInterval = setInterval(captureAndDetect, 500); // 2 FPS
}

function stopRealtime() {
  isRealtime = false;
  if (realtimeInterval) {
    clearInterval(realtimeInterval);
    realtimeInterval = null;
  }

  // UI Reset
  btnCapture.style.display = "flex";
  btnRealtime.style.display = "flex";
  btnStopRealtime.style.display = "none";

  canvas.style.display = "none";

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

async function captureAndDetect() {
  if (!stream || !isRealtime) return;

  // Create temp canvas to capture frame
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = videoCamera.videoWidth;
  tempCanvas.height = videoCamera.videoHeight;
  const ctx = tempCanvas.getContext('2d');

  // Draw current frame
  ctx.drawImage(videoCamera, 0, 0);

  // Convert to blob and send
  tempCanvas.toBlob((blob) => {
    if (blob) {
      sendRealtimeFrame(blob);
    }
  }, 'image/jpeg', 0.8);
}

async function sendRealtimeFrame(blob) {
  const formData = new FormData();
  formData.append("image", blob, "frame.jpg");

  try {
    const response = await fetch("/predict-detection", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) return;

    const data = await response.json();

    if (!isRealtime) return;

    drawRealtimeBoxes(data.detections);

    // ===============================
    // 🔥 TAMBAHAN UPDATE PANEL HASIL
    // ===============================
    if (data.success && data.detections.length > 0) {
      resultBox.style.display = "block";
      initialStateBox.style.display = "none";

      resultValue.innerText = `${data.detections.length} Pisang Terdeteksi`;

      // Ambil label pertama (atau bisa diringkas semua)
      const labels = data.detections.map((d) => d.label);
      const uniqueLabels = [...new Set(labels)];

      additionalInfo.innerText = uniqueLabels.join(", ");
    } else {
      resultValue.innerText = "Tidak Ada Deteksi";
      additionalInfo.innerText = "Model tidak menemukan objek.";
    }
  } catch (err) {
    console.error("Realtime error:", err);
  }
}

function drawRealtimeBoxes(detections) {
  if (!detections || detections.length === 0) return;

  canvas.width = videoCamera.videoWidth;
  canvas.height = videoCamera.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  detections.forEach((det) => {
    if (det.score >= 0.5) {
      const x1 = det.box[0];
      const y1 = det.box[1];
      const x2 = det.box[2];
      const y2 = det.box[3];

      // MIRROR KOORDINAT X
      const mirroredX = canvas.width - x2;
      const width = x2 - x1;
      const height = y2 - y1;

      ctx.strokeStyle = "#00FF00";
      ctx.lineWidth = 4;
      ctx.strokeRect(mirroredX, y1, width, height);

      ctx.fillStyle = "#00FF00";
      ctx.font = "20px Arial";
      const displayLabel = getMaturityLabel(det.class_id);
      ctx.fillText(displayLabel, mirroredX, y1 > 25 ? y1 - 10 : y1 + 25);
    }
  });
}



function captureImage() {
  const context = canvas.getContext("2d");

  canvas.width = videoCamera.videoWidth;
  canvas.height = videoCamera.videoHeight;

  context.drawImage(videoCamera, 0, 0, canvas.width, canvas.height);

  stopCamera();

  videoWrapper.style.display = "none"; // Hide video
  imagePreview.src = canvas.toDataURL("image/png");
  imagePreview.style.display = "block";

  canvas.style.display = "none"; // Hide canvas pending result
  canvas.style.position = 'static'; // Reset position logic for static result
  canvas.style.transform = 'none'; // Reset transform

  calculateResult(imagePreview);
}



// ===============================
// AI PROCESSING
// ===============================

function calculateResult(sourceImg) {
  initialStateBox.style.display = "none";
  resultBox.style.display = "block";
  resultValue.innerText = "Menganalisis...";
  additionalInfo.innerText = "Mohon tunggu...";

  if (currentMode === "klasifikasi") {
    sendToClassificationAPI(sourceImg);
  } else {
    sendToDetectionAPI(sourceImg);
  }
}


// ===============================
// DETECTION API CALL (FLASK)
// ===============================

async function sendToDetectionAPI(sourceImg) {

  try {

    // Convert image preview ke blob
    const blob = await fetch(sourceImg.src).then(r => r.blob());

    const formData = new FormData();
    formData.append("image", blob);  // HARUS SAMA dengan Flask

    console.log("📤 Sending image to backend...");

    const response = await fetch("/predict-detection", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error("Server error");
    }

    const data = await response.json();
    console.log("📥 Response from backend:", data);

    if (!data.success) {
      resultValue.innerText = "Tidak Ada Deteksi";
      additionalInfo.innerText = "Model tidak menemukan objek.";
      return;
    }

    // ===============================
    // TAMPILKAN HASIL
    // ===============================

    console.log(`✅ Found ${data.detections.length} detection(s)`);

    imagePreview.style.display = 'none';

    // Show Video Wrapper to contain the canvas
    if (videoWrapper) videoWrapper.style.display = 'block';
    if (videoCamera) videoCamera.style.display = 'none'; // Hide video in static mode

    if (canvas) {
      canvas.style.display = 'block';
      canvas.style.position = 'static'; // Use static flow for result
      canvas.style.transform = 'none'; // Ensure no mirror for static image
      canvas.style.width = '100%'; // Full width
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const img = new Image();
    img.src = sourceImg.src;

    img.onload = () => {

      canvas.width = img.width;
      canvas.height = img.height;

      ctx.drawImage(img, 0, 0);

      console.log(`🎨 Drawing on canvas (${canvas.width}x${canvas.height})`);

      let count = 0;

      data.detections.forEach(det => {

        if (det.score >= 0.5) {

          count++;

          console.log(`Drawing box for ${det.label} at [${det.box}]`);

          ctx.strokeStyle = "#00FF00";
          ctx.lineWidth = 4;

          ctx.strokeRect(
            det.box[0],
            det.box[1],
            det.box[2] - det.box[0],
            det.box[3] - det.box[1]
          );

          ctx.fillStyle = "#00FF00";
          ctx.font = "18px Arial";

          ctx.fillText(
            `${det.label} ${(det.score * 100).toFixed(1)}%`,
            det.box[0],
            det.box[1] > 20 ? det.box[1] - 10 : det.box[1] + 20
          );
        }
      });

      if (count === 0) {
        resultValue.innerText = "Tidak Ada Pisang";
        resultValue.style.color = "#333";
        additionalInfo.innerText = "Confidence dibawah threshold.";
      } else {

        // JIKA MODE KLASIFIKASI: Tampilkan label dominan saja
        if (currentMode === 'klasifikasi') {
          // Cari deteksi dengan score tertinggi
          let bestDet = data.detections.reduce((prev, current) => (prev.score > current.score) ? prev : current);

          resultValue.innerText = bestDet.label;

          // Set warna berdasarkan label
          if (bestDet.label.includes("Mentah")) resultValue.style.color = "#4caf50";
          else if (bestDet.label.includes("Terlalu")) resultValue.style.color = "#795548";
          else resultValue.style.color = "#ffcc00"; // Matang

          additionalInfo.innerText = `Confidence: ${(bestDet.score * 100).toFixed(1)}%`;

          // Gambar kotak HANYA pada pisang terbaik (opsional, bisa dihilangkan jika mau bersih)
          // Di sini kita gambar kotak tetap untuk visualisasi
          ctx.strokeStyle = resultValue.style.color;
          ctx.lineWidth = 5;
          ctx.strokeRect(
            bestDet.box[0], bestDet.box[1],
            bestDet.box[2] - bestDet.box[0],
            bestDet.box[3] - bestDet.box[1]
          );

        } else {
          // MODE DETEKSI: Tampilkan jumlah
          resultValue.innerText = `${count} Pisang Terdeteksi`;
          resultValue.style.color = "#333";
          additionalInfo.innerText = "Deteksi berbasis SSD MobileNet";
        }
      }
    };

  } catch (error) {

    resultValue.innerText = "Error";
    additionalInfo.innerText = "Gagal terhubung ke server.";
    console.error("❌ Error:", error);
  }

}

// ===============================
// CLASSIFICATION API CALL
// ===============================

async function sendToClassificationAPI(sourceImg) {

  try {

    const blob = await fetch(sourceImg.src).then(r => r.blob());

    const formData = new FormData();
    formData.append("image", blob);

    console.log("📤 Sending image to classification backend...");

    const response = await fetch("/predict-classification", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error("Server error");
    }

    const data = await response.json();
    console.log("📥 Classification response:", data);

    if (!data.success) {
      resultValue.innerText = "Error";
      additionalInfo.innerText = data.error;
      return;
    }

    resultValue.innerText = data.label;

    // Warna berdasarkan label
    if (data.label.includes("Mentah")) {
      resultValue.style.color = "#4caf50";
    } else if (data.label.includes("Terlalu")) {
      resultValue.style.color = "#795548";
    } else {
      resultValue.style.color = "#ffcc00";
    }

    additionalInfo.innerText =
      "Confidence: " + (data.confidence * 100).toFixed(1) + "%";

  } catch (error) {

    resultValue.innerText = "Error";
    additionalInfo.innerText = "Gagal terhubung ke server.";
    console.error("❌ Classification Error:", error);
  }
}
