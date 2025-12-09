document.getElementById("refreshButton").onclick = async () => {
    updateStatus();
};

async function updateStatus() {
    try {
        const response = await fetch("/status");
        const data = await response.json();
        document.getElementById("statusText").innerText = data.status;
    } catch (err) {
        document.getElementById("statusText").innerText = "Unable to fetch status.";
    }
}

updateStatus();

async function sendAction(action) {
    try {
        const response = await fetch(`/action/${action}`);
        const data = await response.json();
        alert(data.message);
    } catch {
        alert("Error sending action.");
    }
}