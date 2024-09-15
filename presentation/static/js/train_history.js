document.addEventListener('DOMContentLoaded', function() {

    get_history();
});

function get_history()
{
    const train_model = document.getElementById('trainModelSelect').value;

    showSpinner();

    debugger;

    fetch(`/training/get_history?start=0&length=10&trainModel=${train_model}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {

            $('#trainHistoryTable').DataTable({
                data: data.data,
                destroy: true,
                serverSide: true,
                ajax: {
                    url: '/training/get_history', // The URL to fetch data from
                    type: 'GET', // HTTP method (GET or POST)
                    data: function (d) {
                        d.train_model = train_model;
                    },
                    error: function (xhr, error, thrown) {
                        console.error('Error fetching data from the server:', error);
                    }
                },
                columnDefs: [
                    {
                        targets: -1, // Target the last column (where buttons are)
                        width: "120px" // Set the width (adjust as necessary)
                    }
                ],
                columns: [
                    { data: 'id' },
                    { data: 'train_model' },
                    { data: 'f1_score' },
                    { data: 'accuracy' },
                    { data: 'loss' },
                    { data: 'auc' },
                    { data: 'aupr' },
                    { data: 'execute_time' },
                    {
                        data: null,
                        orderable: false,
                        render: function (data, type, row) {
                            return `
                                <button class="btn btn-primary" onclick="showplots(${row.id})"><i class="bi bi-bar-chart"></i></button>
                                <button class="btn btn-success" onclick="showconditions(${row.id})"><i class="bi bi-file-earmark-code"></i></button>
                                <button class="btn btn-info" onclick="showdetails(${row.id})"><i class="bi bi-info-circle"></i></button>
                            `;
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

function showdetails(id) {
    window.location.href = `/training/training_history_details/${id}`;
}

function showplots(id) {
    window.location.href = `/training/training_history_plots/${id}`;
}
