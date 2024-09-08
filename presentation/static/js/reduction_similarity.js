function filterReductionSimilarities(category)
{
    let similarity_type = '';
    let reduction_category = '';

    try {
        similarity_type = document.getElementById('similaritySelect').value;
        reduction_category = document.querySelector('input[name="reductionRadioOption"]:checked').value;

        if (similarity_type === '') {
            throw new Error('Text type is required');
        }
    } catch (ex){
        alert(`Please select the input fields! \n ${ex}`)
        return false;
    }

    showSpinner();

    fetch(`/reduction/similarity?start=0&length=10&reduction_category=${reduction_category}&category=${category}&similarity_type=${similarity_type}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {

            $('#reductionTable').DataTable({
                data: data.data,
                destroy: true,
                serverSide: true,
                ajax: {
                    url: '/reduction/similarity', // The URL to fetch data from
                    type: 'GET', // HTTP method (GET or POST)
                    data: function (d) {
                        d.category = category;
                        d.similarity_type = similarity_type;
                        d.reduction_category = reduction_category;
                    },
                    error: function (xhr, error, thrown) {
                        console.error('Error fetching data from the server:', error);
                    }
                },
                columns: [
                    { data: 'id' },
                    { data: 'drugbank_id' },
                    {
                        data: 'reduction_value',
                        className: 'eighty-percent-width',
                        render: function (data, type, row) {
                            if (type === 'display') {
                                return data.length > 500 ? data.substr(0, 500) + '...' : data;
                            }
                            return data; // Return the full value for other data types like export or sorting
                        }
                    }
                ],
                paging: true,
                searching: true,
                ordering: true,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                language: {
                    search: "_INPUT_",
                    searchPlaceholder: "Search ..."
                }
            });
        } else {
            console.log('Error: No data found.');
        }

        hideSpinner(true);
    })
    .catch(error => {
        console.log('Error:', error);
        hideSpinner(false);
    });
}

function calculateReductionSimilarities(category)
{
    let similarity_type = '';
    let reduction_category = '';

    try {
        similarity_type = document.getElementById('similaritySelect').value;
        reduction_category = document.querySelector('input[name="reductionRadioOption"]:checked').value;

        if (similarity_type === '') {
            throw new Error('Text type is required');
        }
    } catch (ex){
        alert(`Please select the input fields! \n ${ex}`)
        return false;
    }

    showSpinner();

    fetch(`/reduction/calculateSimilarity?reduction_category=${reduction_category}&category=${category}&similarity_type=${similarity_type}`, {
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
        console.log('Error:', error);
        hideSpinner(false);
    });
}
