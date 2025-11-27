function initAudioPage() {
let mediaRecorder;
let chunks = [];
let audioFile = null;

const recordBtn = document.getElementById("recordBtn");
const uploadInput = document.getElementById("audioUpload");
const processBtn = document.getElementById("processBtn");
const transcriptEl = document.getElementById("transcript");
const signVideo = document.getElementById("signVideo");
const placeholder = document.getElementById("noAnimation");

recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    recordBtn.textContent = "🎙️ Record Speech";
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.onstop = () => {
    // const blob = new Blob(chunks, { type: "audio/wav" });
    const blob = new Blob(chunks, { type: "audio/webm" });

    audioFile = blob;
    chunks = [];
  };

  mediaRecorder.start();
  recordBtn.textContent = "⏹️ Stop Recording";
});

uploadInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    audioFile = file;
    alert("Audio file selected!");
  }
});

processBtn.addEventListener("click", async () => {
  if (!audioFile) {
    alert("Please record or upload audio first!");
    return;
  }

  transcriptEl.textContent = "Processing...";
  placeholder.style.display = "block";
  signVideo.style.display = "none";

  const formData = new FormData();
  formData.append("file", audioFile);

  try {
    console.log("Sending audio file to API...");
    const response = await fetch("http://localhost:8000/transcribe/", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    console.log("API response:", data);
    // console.log("hello");


    // ✅ Update UI after getting response
    transcriptEl.textContent = data.text || "No text recognized.";
    signVideo.src = "http://localhost:8000"+data.video_url; 
    signVideo.load();
signVideo.play();
    // Replace with your real animation output
    signVideo.style.display = "block";
    placeholder.style.display = "none";
  } catch (err) {
    console.log("errorrrr̥");
    console.error("Error:", err);
    transcriptEl.textContent = "Failed to process audio.";
  }
});
}
// function initAudioPage() {

//     console.log("Audio page initialized");

//     let mediaRecorder;
//     let chunks = [];
//     let audioFile = null;

//     const recordBtn = document.getElementById("recordBtn");
//     const uploadInput = document.getElementById("audioUpload");
//     const processBtn = document.getElementById("processBtn");
//     const transcriptEl = document.getElementById("transcript");
//     const signVideo = document.getElementById("signVideo");

//     // --- Record button ---
//     recordBtn.onclick = async () => {

//         if (mediaRecorder && mediaRecorder.state === "recording") {
//             mediaRecorder.stop();
//             recordBtn.textContent = "🎙️ Record Speech";
//             return;
//         }

//         const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

//         mediaRecorder = new MediaRecorder(stream);
//         chunks = [];

//         mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

//         mediaRecorder.onstop = async () => {
//             const blob = new Blob(chunks, { type: "audio/webm" });
//             audioFile = new File([blob], "recorded_audio.webm", { type: "audio/webm" });
//         };

//         mediaRecorder.start();
//         recordBtn.textContent = "⏹ Stop Recording";
//     };

//     // --- Upload input ---
//     uploadInput.onchange = (e) => {
//         audioFile = e.target.files[0];
//     };

//     // --- Process button ---
//     processBtn.onclick = async () => {
//         if (!audioFile) {
//             alert("Please record or upload audio first.");
//             return;
//         }

//         transcriptEl.textContent = "Processing...";
        
//         const formData = new FormData();
//         formData.append("file", audioFile);

//         let res = await fetch("http://localhost:8000/transcribe/", {
//             method: "POST",
//             body: formData
//         });

//         const data = await res.json();
//         transcriptEl.textContent = data.text;

//         signVideo.src = "http://localhost:8000" + data.video_url;
//         signVideo.load();
//         signVideo.play();
//     };
// }

// // Run initializer
initAudioPage();
