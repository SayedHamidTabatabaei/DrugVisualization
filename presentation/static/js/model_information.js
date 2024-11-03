function showmodelinfo(id) {

    fetch(`/training/get_history_model_information?trainHistoryId=${id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then(response => response.json())
        .then(data => {
            if (data.status) {
                const parsedData = typeof data.data === 'string' ? JSON.parse(data.data) : data.data;  // Parse only if it's a string
                if (parsedData.length > 1) {
                    let htmlContent = "";

                    parsedData.forEach((item, index) => {
                      htmlContent += `<strong style='font-size: 250%; font-weight: bold'>k: ${index}</strong><pre style='font-size: 150%; font-weight: bold'> ${item['model_summary']}</pre><br><hr><br>`;
                    });

                    document.getElementById('jsonData').innerHTML = htmlContent;
                }
                else {
                    document.getElementById('jsonData').innerHTML = "<pre style='font-size: 150%; font-weight: bold'>" + parsedData['model_summary'] + "</pre>";
                }

                document.getElementById('jsonModal').style.display = 'block';
            } else {
                console.log('Error: No data found.');
            }
        })
        .catch(error => console.log('Error:', error));
}

function closeModal() {
    document.getElementById('jsonModal').style.display = 'none';
}

function summaryToHtml(summaryText) {
    // Split text by lines
    const lines = summaryText.trim().split('\n');

    // Start the table HTML
    let html = "<table><tr>";

    // First line as table headers
    const headers = lines[0].split(/\s{2,}/);  // Split by 2 or more spaces
    headers.forEach(header => {
        html += `<th>${header.trim()}</th>`;
    });
    html += "</tr>";

    // Loop over the rest of the lines for table rows
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === "") continue;  // Skip empty lines
        const cols = lines[i].split(/\s{2,}/); // Split by 2 or more spaces
        html += "<tr>";
        cols.forEach(col => {
            html += `<td>${col.trim()}</td>`;
        });
        html += "</tr>";
    }
    html += "</table>";

    return html;
}
