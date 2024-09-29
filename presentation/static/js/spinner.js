function showSpinner() {
    document.getElementById('loadingSpinner').style.display = 'block';
    createStatusStrip()
}

function hideSpinner(ok) {
    document.getElementById('loadingSpinner').style.display = 'none';
    removeStatusStrip(ok)
}

function createStatusStrip() {
    // Dynamically create the status strip
    const statusStrip = document.createElement('div');
    statusStrip.classList.add('statusStrip');
    statusStrip.id = 'statusStrip';
    document.body.appendChild(statusStrip);

    // Set initial color to yellow with transition
    statusStrip.style.backgroundColor = 'yellow';

    return statusStrip;
}

function removeStatusStrip(ok) {
    // Select the status strip
    const statusStrip = document.getElementById('statusStrip');
    if (statusStrip) {

        if (ok) {
            statusStrip.style.backgroundColor = 'green';
        }
        else{
            statusStrip.style.backgroundColor = 'red';
        }

        // Fade out the strip after 2 seconds
        setTimeout(() => {
            statusStrip.style.opacity = '0';  // Fades the strip out

            // Remove the element after the fade-out animation
            setTimeout(() => {
                document.body.removeChild(statusStrip);
            }, 500);  // Time for opacity transition
        }, 1000);  // Wait before starting fade out
    }
}
