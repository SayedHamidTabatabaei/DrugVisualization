function showconditions(id) {

    fetch(`/training/get_history_conditions?trainHistoryId=${id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then(response => response.json())
        .then(data => {
            if (data.status) {
                let formattedJson = '';
                const parsedData = typeof data.data === 'string' ? JSON.parse(data.data) : data.data;  // Parse only if it's a string

                for (let key in parsedData) {
                    if (parsedData.hasOwnProperty(key) && parsedData[key]) {  // Check parsedData's properties, not data.data
                        formattedJson += `<strong>${key}</strong>: ${parsedData[key]}<br>`;
                    }
                }

                document.getElementById('jsonData').innerHTML = formattedJson;

                document.getElementById('jsonModal').style.display = 'block';
            } else {
                console.log('Error: No data found.');
            }
        })
        .catch(error => console.log('Error:', error));
}

// Function to close the modal
function closeModal() {
    document.getElementById('jsonModal').style.display = 'none';
}
