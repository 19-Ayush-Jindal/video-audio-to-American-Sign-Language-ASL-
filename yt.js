let ytPlayer = null;
let ws = null;
let firstChunk = false;
let lastBlobUrl = null;
let finalChunks = [];

function onYouTubeIframeAPIReady() {
    console.log("YT API Loaded");
}

function extractVideoID(url) {
    const patterns = [
        /v=([^&]+)/,
        /youtu\.be\/([^?]+)/,
        /embed\/([^?]+)/,
    ];
    for (const p of patterns) {
        const match = url.match(p);
        if (match) return match[1];
    }
    return null;
}

function start() {
    const url = document.getElementById("ytUrl").value;
    const videoId = extractVideoID(url);

    if (!videoId) {
        alert("Invalid YouTube URL");
        return;
    }

    document.getElementById("ytPlayer").style.display = "block";

    document.getElementById("ytPlayer").src =
        `https://www.youtube.com/embed/${videoId}?enablejsapi=1&mute=1&playsinline=1&autoplay=0`;

    ytPlayer = new YT.Player("ytPlayer", {
        events: {
            "onReady": () => console.log("YT Player READY")
        }
    });

    ws = new WebSocket("ws://localhost:8001/stream_asl");

    ws.onopen = () => {
        console.log("WS connected");
        ws.send(url);
    };

    ws.onmessage = event => {
        console.log("Received ASL chunk");

        if (event.data === "DONE") {
            document.getElementById("downloadLink").style.display = "block";
        }

        const blob = new Blob([event.data], { type: "video/mp4" });
        finalChunks.push(blob);

        if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
        lastBlobUrl = URL.createObjectURL(blob);

        const video = document.getElementById("signPlayer");
        video.src = lastBlobUrl;
        video.load();
        video.play();

        if (!firstChunk) {
            firstChunk = true;
            setTimeout(syncYouTube, 800);
        }
    };

    ws.onerror = err => console.error("WebSocket error:", err);
}

function syncYouTube() {
    if (ytPlayer && ytPlayer.playVideo) {
        ytPlayer.playVideo();
    } else {
        setTimeout(syncYouTube, 300);
    }
}

function fast() {
    document.getElementById("signPlayer").playbackRate = 2.0;
}

document.addEventListener("click", (e) => {
    if (e.target.id === "startBtn") start();
});
