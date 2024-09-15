document.addEventListener('DOMContentLoaded', function() {

    const train_history_id = document.getElementById('train-history-id').textContent;

    showSpinner();

    fetch(`/training/get_history_plots?trainHistoryId=${train_history_id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            const plotsDiv = document.getElementById('plots');

            plotsDiv.innerHTML = '';

            data.data.forEach(image => {
                const imgElement = document.createElement('img');
                imgElement.src = image.path;  // Set the image URL
                imgElement.alt = image.name;  // Set alt text for accessibility
                imgElement.style.maxWidth = '100%';  // Optional: make sure image fits within the div

                plotsDiv.appendChild(imgElement);
            });

        } else {
            console.log('Error: No data found.');
        }
        hideSpinner(true);
    })
    .catch(error => {
        console.log('Error:', error)
        hideSpinner(false);
    });
});