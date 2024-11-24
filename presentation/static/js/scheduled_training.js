document.addEventListener('DOMContentLoaded', function() {

    get_schedules();
});

function get_schedules()
{
    const train_model = document.getElementById('trainModelSelect').value;

    showSpinner();

    fetch(`/training/get_schedules?trainModel=${train_model}`, {
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
                columns: [
                    { data: 'id' },
                    { data: 'train_model' },
                    { data: 'name' },
                    { data: 'description' },
                    { data: 'min_sample_count' },
                    { data: 'schedule_date' },
                    {
                        data: null,
                        orderable: false,
                        render: function (data, type, row) {
                            return `
                                <button class="btn btn-danger" onclick="schedule_delete(${row.id})"><i class="bi bi-trash"></i></button>
                                <button class="btn btn-info" onclick="run_train(${row.id})"><i class="bi bi-play-fill"></i></button>
                            `;
                        }
                    }
                ],
                paging: false,
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

function schedule_delete(id) {

    showSpinner();

    fetch(`/training/training_schedule_delete?id=${id}`, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            get_schedules();
            hideSpinner(true);

        } else {
            console.log('Error: No data found.');
            hideSpinner(false);
        }
    })
    .catch(error => {
        console.log('Error:', error)
        hideSpinner(false);
    });
}

function run_train(id) {

    showSpinner();

    fetch(`/training/run_train?id=${id}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            get_schedules();
            hideSpinner(true);

        } else {
            console.log('Error: No data found.');
            hideSpinner(false);
        }
    })
    .catch(error => {
        console.log('Error:', error)
        hideSpinner(false);
    });
}

function add_schedule_training() {
    window.location.href = `/training/train`;
}
