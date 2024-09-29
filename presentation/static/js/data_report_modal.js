function showdatareport(id) {

    fetch(`/training/get_history_data_reports?trainHistoryId=${id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then(response => response.json())
        .then(data => {
            debugger;
            if (data.status) {
                let formattedJson = '';
                const parsedData = typeof data.data === 'string' ? JSON.parse(data.data) : data.data;  // Parse only if it's a string

                for (let key in parsedData) {
                    if (parsedData.hasOwnProperty(key) && parsedData[key]) {
                        if (Array.isArray(parsedData[key])) {
                            // If it's an array, iterate over each object in the array
                            formattedJson += `<strong>${key}</strong>: [`;
                            parsedData[key].forEach((obj, index) => {
                                if (typeof obj === 'object') {
                                    formattedJson += `&nbsp;{`;
                                    for (let prop in obj) {
                                        if (obj.hasOwnProperty(prop)) {
                                            formattedJson += `&nbsp;<strong>${prop}</strong>: ${obj[prop]}`;
                                        }
                                    }
                                    formattedJson += `&nbsp;}`;
                                } else {
                                    formattedJson += `&nbsp;&nbsp;&nbsp;&nbsp;${obj}<br>`;
                                }
                            });
                            formattedJson += `]<br><br>`;
                        } else if (typeof parsedData[key] === 'object') {
                            // If it's a single object, print its properties
                            formattedJson += `<strong>${key}</strong>: {<br>`;
                            for (let prop in parsedData[key]) {
                                if (parsedData[key].hasOwnProperty(prop)) {
                                    formattedJson += `&nbsp;&nbsp;&nbsp;&nbsp;<strong>${prop}</strong>: ${parsedData[key][prop]}<br>`;
                                }
                            }
                            formattedJson += `}<br><br>`;
                        } else {
                            // For primitive values, just display them
                            formattedJson += `<strong>${key}</strong>: ${parsedData[key]}<br><br>`;
                        }
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
