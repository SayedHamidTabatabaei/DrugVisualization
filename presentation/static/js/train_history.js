document.addEventListener('DOMContentLoaded', function() {

    get_history();
});

function get_history()
{
    const train_model = document.getElementById('trainModelSelect').value;

    showSpinner();

    fetch(`/training/get_all_history?start=0&length=10&trainModel=${train_model}`, {
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
                    url: '/training/get_all_history', // The URL to fetch data from
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
                    { data: 'name' },
                    { data: 'description' },
                    // { data: 'execute_time' },
                    {
                        data: null,
                        orderable: false,
                        render: function (data, type, row) {
                            return `
                                <div style="min-width: 100px;">
                                    <button style="margin:1px;" class="btn btn-danger" onclick="removeRow(${row.id})" data-bs-toggle="tooltip" title="Remove"><i class="bi bi-trash"></i></button>
                                    <button style="margin:1px;" class="btn btn-primary" onclick="showplots(${row.id})" data-bs-toggle="tooltip" title="Show Plots"><i class="bi bi-bar-chart"></i></button>
                                    <button style="margin:1px;" class="btn btn-info" onclick="showdetails(${row.id})" data-bs-toggle="tooltip" title="Show Details"><i class="bi bi-info-circle"></i></button>
                                    <button style="margin:1px;" class="btn btn-success" onclick="showconditions(${row.id})" data-bs-toggle="tooltip" title="Show Conditions"><i class="bi bi-file-earmark-code"></i></button>
                                    <button style="margin:1px;" class="btn btn-success" onclick="showdatareport(${row.id})" data-bs-toggle="tooltip" title="Show Data Report"><i class="bi bi-file-earmark-code"></i></button>
                                </div>
                            `;
                        }
                    }
                ],
                rowCallback: function(row, data) {
                    if (!data.is_completed) { // Check if is_completed is false
                        $(row).css('background-color', 'red'); // Set the row background color to red
                    }
                },
                paging: true,
                searching: true,
                ordering: true,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                language: {
                    search: "_INPUT_",
                    searchPlaceholder: "Search ..."
                },
                initComplete: function() {
                    let table = document.getElementById('trainHistoryTable');
                    if (table) {
                        table.removeAttribute('style');
                    }
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

function removeRow(id) {
    if (confirm("Are you sure you want to remove this row?")) {
        // Example: Make an AJAX request to remove the row from the server
        $.ajax({
            url: `/training/training_history_remove/${id}`, // URL to handle the remove action on the server
            type: 'POST',
            success: function(response) {
                if (response.success) {
                    // Remove the row from the DataTable
                    const table = $('#trainHistoryTable').DataTable();
                    table.row(`#row_${id}`).remove().draw(); // Ensure that you have a unique row identifier for this to work
                    alert("Row removed successfully");
                } else {
                    alert("Error removing row");
                }
            },
            error: function(xhr, status, error) {
                console.error("Error removing row:", error);
                alert("Error removing row");
            }
        });
    }
}
