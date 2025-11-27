// Load default Home page
window.onload = () => {
    loadPage("home.html");
};

function loadPage(page) {
    fetch(page)
        .then(res => res.text())
        .then(html => {
            document.getElementById("content").innerHTML = html;

            if (page === "yt.html") {
                // Load YT API
                const ytApi = document.createElement("script");
                ytApi.src = "https://www.youtube.com/iframe_api";
                document.body.appendChild(ytApi);

                // Load your YT Logic
                const ytScript = document.createElement("script");
                ytScript.src = "yt.js";
                document.body.appendChild(ytScript);
            }
            if (page === "audio.html") {

    // Remove previous audio script if exists
    document.querySelectorAll("script[data-audio]").forEach(s => s.remove());

    // Load a fresh script every time
    const audioScript = document.createElement("script");
    audioScript.src = "audio.js?v=" + Date.now();   // cache-bust
    audioScript.dataset.audio = "true";
    document.body.appendChild(audioScript);
}

        });
}


// Handle navbar clicks
document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => {
        loadPage(btn.dataset.page);
    });
});

// Home icon
document.getElementById("home-btn").onclick = () => {
    loadPage("home.html");
};
