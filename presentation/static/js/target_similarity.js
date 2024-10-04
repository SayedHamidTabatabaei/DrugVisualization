function filterTargetSimilarity() {

    let similarityType = '';

    try {
        similarityType = document.getElementById('similaritySelect').value;
        if (similarityType === '') {
            throw new Error('Similarity is required');
        }
    } catch (ex){
        alert(`Please select the input fields! \n ${ex}`)
        return false;
    }

    showSpinner();

    fetch(`/target/targetSimilarity?start=0&length=10&similarityType=${similarityType}`)
        .then(response => response.json())
        .then(data => {

            if ($.fn.DataTable.isDataTable('#target_similarity_table')) {
                $('#target_similarity_table').DataTable().clear().destroy();
            }

            let isFirstTime = true;

            const tableHeaders = document.getElementById('tableHeaders');
            tableHeaders.innerHTML = '';

            data.columns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                tableHeaders.appendChild(th);
            });

            $('#target_similarity_table').DataTable({
                scrollX: true,
                scrollY: 400,
                fixedColumns: {
                    leftColumns: 4,
                },
                serverSide: true,
                ajax: function (funcData, callback, settings) {

                    if(!isFirstTime)
                    {
                        const params = {
                            start: funcData.start,
                            length: funcData.length,
                            draw: funcData.draw
                        };
                        fetch(`/target/targetSimilarity?start=${params.start}&length=${params.length}&draw=${params.draw}&similarityType=${similarityType}`)
                            .then(response => response.json())
                            .then(data => {
                                callback(data);
                            })
                            .catch(error => console.log('Error fetching data:', error));
                    }
                    else
                    {
                        callback(data);
                    }

                    isFirstTime = false;
                },
                columns: data.columns ? data.columns.map(col => ({data: col})) : [],
                paging: true,
                ordering: true,
                lengthMenu: [[10, 25, 50, 100000], [10, 25, 50, "All"]],
                initComplete: function() {
                    let table = document.getElementById('target_similarity_table');
                    if (table) {
                        table.removeAttribute('style');
                    }
                }
            });

            hideSpinner(true);
        })
        .catch(error => {
            console.log('Error fetching data:', error);
            hideSpinner(false);
        });
}

function calculateTargetSimilarity()
{
    let similarityType = '';

    try {
        similarityType = document.getElementById('similaritySelect').value;
        if (similarityType === '') {
            throw new Error('Similarity is required');
        }
    } catch (ex){
        alert(`Please select the input fields! \n ${ex}`)
        return false;
    }

    showSpinner();

    fetch(`/target/calculateTargetSimilarity?similarityType=${similarityType}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        console.log('Success:', data);
        hideSpinner(true);
    })
    .catch(error => {
        console.error('Error:', error);
        hideSpinner(false);
    });
}
