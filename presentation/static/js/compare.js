document.addEventListener('DOMContentLoaded', function() {

    get_history();
});

function get_history()
{
    const train_model = document.getElementById('trainModelSelect').value;

    showSpinner();

    fetch(`/training/get_history?start=0&length=10000&trainModel=${train_model}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {

            const maxF1Score = Math.max(...data.data.map(item => item.f1_score));
            const maxAccuracy = Math.max(...data.data.map(item => item.accuracy));
            const minLoss = Math.min(...data.data.map(item => item.loss));
            const maxAuc = Math.max(...data.data.map(item => item.auc));
            const maxAupr = Math.max(...data.data.map(item => item.aupr));
            const maxRecall = Math.max(...data.data.map(item => item.recall));
            const maxPrecision = Math.max(...data.data.map(item => item.precision));

            $('#selectableTrainTable').DataTable({
                data: data.data,
                destroy: true,
                paging: false,
                columns: [
                    {
                        data: null,
                        render: function(data, type, row) {
                            return '<label class="checkbox-label" style="width: 30px;">\n' +
                                '    <input type="checkbox" id="target-check" onclick="compare_trainings()" value="' + row.id + '">\n' +
                                '    <span class="checkbox-custom" style="margin-top: 0;"></span>' +
                                '</label>'
                        },
                        orderable: false,
                        searchable: false
                    },
                    { data: 'id' },
                    { data: 'train_model' },
                    {
                        data: 'f1_score',
                        render: function(data, type, row) {
                            if (row.f1_score === maxF1Score) {
                                return '<strong>' + row.f1_score + '</strong>';
                            } else {
                                return row.f1_score;
                            }
                        }
                    },
                    {
                        data: 'accuracy',
                        render: function(data, type, row) {
                            if (row.accuracy === maxAccuracy) {
                                return '<strong>' + row.accuracy + '</strong>';
                            } else {
                                return row.accuracy;
                            }
                        }
                    },
                    {
                        data: 'loss',
                        render: function(data, type, row) {
                            if (row.loss === minLoss) {
                                return '<strong>' + row.loss + '</strong>';
                            } else {
                                return row.loss;
                            }
                        }
                    },
                    {
                        data: 'auc',
                        render: function(data, type, row) {
                            if (row.auc === maxAuc) {
                                return '<strong>' + row.auc + '</strong>';
                            } else {
                                return row.auc;
                            }
                        }
                    },
                    {
                        data: 'aupr',
                        render: function(data, type, row) {
                            if (row.aupr === maxAupr) {
                                return '<strong>' + row.aupr + '</strong>';
                            } else {
                                return row.aupr;
                            }
                        }
                    },
                    {
                        data: 'recall',
                        render: function(data, type, row) {
                            if (row.recall === maxRecall) {
                                return '<strong>' + row.recall + '</strong>';
                            } else {
                                return row.recall;
                            }
                        }
                    },
                    {
                        data: 'precision',
                        render: function(data, type, row) {
                            if (row.precision === maxPrecision) {
                                return '<strong>' + row.precision + '</strong>';
                            } else {
                                return row.precision;
                            }
                        }
                    },
                    { data: 'execute_time' }
                ],
                searching: true,
                ordering: true,
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

function compare_trainings()
{
    const activeTab = document.querySelector('.tab.active');
    activeTab.click();
}

function openTab(evt, tabName) {
    // Get all elements with class="tab-content" and hide them
    const tabContents = document.getElementsByClassName("tab-content");
    const tabs = document.getElementsByClassName("tab");
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].style.display = "none";
        tabContents[i].classList.remove("active");
        tabs[i].classList.remove("active");
    }

    // Show the current tab, and add an "active" class to the tab and content
    document.getElementById(tabName).style.display = "block";
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

function fill_comparing_plot(comparing_plot_type){

    let selectedIds = [];
    $('input[type="checkbox"]:checked').each(function() {
        selectedIds.push($(this).val()); // Get the value of the checked checkbox
    });

    fetch(`/training/get_comparing_plot?trainHistoryIds=${selectedIds}&ComparePlotType=${comparing_plot_type}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            const plot = document.getElementById(`${comparing_plot_type}-plot`);

            plot.src = `data:image/png;base64,${data.image.path}`;
            plot.alt = data.image.name;
            plot.style.maxWidth = '110%';

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
