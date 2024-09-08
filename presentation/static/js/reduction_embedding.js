function filterReductionEmbeddings()
{
    let text_type = '';
    let embedding_type = '';
    let reduction_category = '';

    try {
        text_type = document.getElementById('propertiesSelect').value;
        embedding_type = document.querySelector('input[name="embeddingRadioOption"]:checked').value;
        reduction_category = document.querySelector('input[name="reductionRadioOption"]:checked').value;

        if (text_type === '') {
            throw new Error('Text type is required');
        }
    } catch (ex){
        alert(`Please select the input fields! \n ${ex}`)
        return false;
    }

    showSpinner();

    fetch(`/reduction/embedding?start=0&length=10&reduction_category=${reduction_category}&text_type=${text_type}&embedding_type=${embedding_type}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {

            $('#embeddingsTable').DataTable({
                data: data.data,
                destroy: true,
                serverSide: true,
                ajax: {
                    url: '/reduction/embedding', // The URL to fetch data from
                    type: 'GET', // HTTP method (GET or POST)
                    data: function (d) {
                        d.text_type = text_type;
                        d.embedding_type = embedding_type;
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
        console.log('Error:', error)
        hideSpinner(false);
    });
}

function calculateReductionEmbeddings()
{
    let text_type = '';
    let embedding_type = '';
    let reduction_category = '';

    try {
        text_type = document.getElementById('propertiesSelect').value;
        embedding_type = document.querySelector('input[name="embeddingRadioOption"]:checked').value;
        reduction_category = document.querySelector('input[name="reductionRadioOption"]:checked').value;

        if (text_type === '') {
            throw new Error('Text type is required');
        }
    } catch (ex){
        alert(`Please select the input fields! \n ${ex}`)
        return false;
    }

    showSpinner();

    fetch(`/reduction/calculateEmbedding?reduction_category=${reduction_category}&embedding_type=${embedding_type}&text_type=${text_type}`, {
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
