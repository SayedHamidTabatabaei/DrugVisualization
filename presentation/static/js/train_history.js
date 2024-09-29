document.addEventListener('DOMContentLoaded', function() {

    get_history();
});

function get_history()
{
    const train_model = document.getElementById('trainModelSelect').value;

    showSpinner();

    fetch(`/training/get_history?start=0&length=10&trainModel=${train_model}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {

            const maxAccuracy = Math.max(...data.data.map(item => item.accuracy));
            const minLoss = Math.min(...data.data.map(item => item.loss));
            const maxF1Score_Weighted = Math.max(...data.data.map(item => item.f1_score_weighted));
            const maxF1Score_Micro = Math.max(...data.data.map(item => item.f1_score_micro));
            const maxF1Score_Macro = Math.max(...data.data.map(item => item.f1_score_macro));
            const maxAuc_Weighted = Math.max(...data.data.map(item => item.auc_weighted));
            const maxAuc_Micro = Math.max(...data.data.map(item => item.auc_micro));
            const maxAuc_Macro = Math.max(...data.data.map(item => item.auc_macro));
            const maxAupr_Weighted = Math.max(...data.data.map(item => item.aupr_weighted));
            const maxAupr_Micro = Math.max(...data.data.map(item => item.aupr_micro));
            const maxAupr_Macro = Math.max(...data.data.map(item => item.aupr_macro));
            const maxRecall_Weighted = Math.max(...data.data.map(item => item.recall_weighted));
            const maxRecall_Micro = Math.max(...data.data.map(item => item.recall_micro));
            const maxRecall_Macro = Math.max(...data.data.map(item => item.recall_macro));
            const maxPrecision_Weighted = Math.max(...data.data.map(item => item.precision_weighted));
            const maxPrecision_Micro = Math.max(...data.data.map(item => item.precision_micro));
            const maxPrecision_Macro = Math.max(...data.data.map(item => item.precision_macro));

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
                    {
                        data: 'accuracy',
                        render: function(data, type, row) {
                            const formatted_accuracy = row.accuracy ? row.accuracy.toFixed(2) : '0.00';

                            if (row.accuracy === maxAccuracy) {

                                return '<strong>' + formatted_accuracy + '</strong>';
                            } else {
                                return formatted_accuracy;
                            }
                        }
                    },
                    {
                        data: 'loss',
                        render: function(data, type, row) {
                            const formatted_loss = row.loss ? row.loss.toFixed(2) : '0.00';

                            if (row.loss === minLoss) {
                                return '<strong>' + formatted_loss + '</strong>';
                            } else {
                                return formatted_loss;
                            }
                        }
                    },
                    {
                        data: 'f1_score_weighted',
                        render: function(data, type, row) {
                            const formatted_f1_score_weighted = row.f1_score_weighted ? row.f1_score_weighted.toFixed(2) : '0.00';

                            if (row.f1_score_weighted === maxF1Score_Weighted) {
                                return '<strong>' + formatted_f1_score_weighted + '</strong>';
                            } else {
                                return formatted_f1_score_weighted;
                            }
                        }
                    },
                    {
                        data: 'f1_score_micro',
                        render: function(data, type, row) {
                            const formatted_f1_score_micro = row.f1_score_micro ? row.f1_score_micro.toFixed(2) : '0.00';

                            if (row.f1_score_micro === maxF1Score_Micro) {
                                return '<strong>' + formatted_f1_score_micro + '</strong>';
                            } else {
                                return formatted_f1_score_micro;
                            }
                        }
                    },
                    {
                        data: 'f1_score_macro',
                        render: function(data, type, row) {
                            const formatted_f1_score_macro = row.f1_score_macro ? row.f1_score_macro.toFixed(2) : '0.00';

                            if (row.f1_score_macro === maxF1Score_Macro) {
                                return '<strong>' + formatted_f1_score_macro + '</strong>';
                            } else {
                                return formatted_f1_score_macro;
                            }
                        }
                    },
                    {
                        data: 'auc_weighted',
                        render: function(data, type, row) {
                            const formatted_auc_weighted = row.auc_weighted ? row.auc_weighted.toFixed(2) : '0.00';

                            if (row.auc_weighted === maxAuc_Weighted) {
                                return '<strong>' + formatted_auc_weighted + '</strong>';
                            } else {
                                return formatted_auc_weighted;
                            }
                        }
                    },
                    {
                        data: 'auc_micro',
                        render: function(data, type, row) {
                            const formatted_auc_micro = row.auc_micro ? row.auc_micro.toFixed(2) : '0.00';

                            if (row.auc_micro === maxAuc_Micro) {
                                return '<strong>' + formatted_auc_micro + '</strong>';
                            } else {
                                return formatted_auc_micro;
                            }
                        }
                    },
                    {
                        data: 'auc_macro',
                        render: function(data, type, row) {
                            const formatted_auc_macro = row.auc_macro ? row.auc_macro.toFixed(2) : '0.00';

                            if (row.auc_macro === maxAuc_Macro) {
                                return '<strong>' + formatted_auc_macro + '</strong>';
                            } else {
                                return formatted_auc_macro;
                            }
                        }
                    },
                    {
                        data: 'aupr_weighted',
                        render: function(data, type, row) {
                            const formatted_aupr_weighted = row.aupr_weighted ? row.aupr_weighted.toFixed(2) : '0.00';

                            if (row.aupr_weighted === maxAupr_Weighted) {
                                return '<strong>' + formatted_aupr_weighted + '</strong>';
                            } else {
                                return formatted_aupr_weighted;
                            }
                        }
                    },
                    {
                        data: 'aupr_micro',
                        render: function(data, type, row) {
                            const formatted_aupr_micro = row.aupr_micro ? row.aupr_micro.toFixed(2) : '0.00';

                            if (row.aupr_micro === maxAupr_Micro) {
                                return '<strong>' + formatted_aupr_micro + '</strong>';
                            } else {
                                return formatted_aupr_micro;
                            }
                        }
                    },
                    {
                        data: 'aupr_macro',
                        render: function(data, type, row) {
                            const formatted_aupr_macro = row.aupr_macro ? row.aupr_macro.toFixed(2) : '0.00';

                            if (row.aupr_macro === maxAupr_Macro) {
                                return '<strong>' + formatted_aupr_macro + '</strong>';
                            } else {
                                return formatted_aupr_macro;
                            }
                        }
                    },
                    {
                        data: 'recall_weighted',
                        render: function(data, type, row) {
                            const formatted_recall_weighted = row.recall_weighted ? row.recall_weighted.toFixed(2) : '0.00';

                            if (row.recall_weighted === maxRecall_Weighted) {
                                return '<strong>' + formatted_recall_weighted + '</strong>';
                            } else {
                                return formatted_recall_weighted;
                            }
                        }
                    },
                    {
                        data: 'recall_micro',
                        render: function(data, type, row) {
                            const formatted_recall_micro = row.recall_micro ? row.recall_micro.toFixed(2) : '0.00';

                            if (row.recall_micro === maxRecall_Micro) {
                                return '<strong>' + formatted_recall_micro + '</strong>';
                            } else {
                                return formatted_recall_micro;
                            }
                        }
                    },
                    {
                        data: 'recall_macro',
                        render: function(data, type, row) {
                            const formatted_recall_macro = row.recall_macro ? row.recall_macro.toFixed(2) : '0.00';

                            if (row.recall_macro === maxRecall_Macro) {
                                return '<strong>' + formatted_recall_macro + '</strong>';
                            } else {
                                return formatted_recall_macro;
                            }
                        }
                    },
                    {
                        data: 'precision_weighted',
                        render: function(data, type, row) {
                            const formatted_precision_weighted = row.precision_weighted ? row.precision_weighted.toFixed(2) : '0.00';

                            if (row.precision_weighted === maxPrecision_Weighted) {
                                return '<strong>' + formatted_precision_weighted + '</strong>';
                            } else {
                                return formatted_precision_weighted;
                            }
                        }
                    },
                    {
                        data: 'precision_micro',
                        render: function(data, type, row) {
                            const formatted_precision_micro = row.precision_micro ? row.precision_micro.toFixed(2) : '0.00';

                            if (row.precision_micro === maxPrecision_Micro) {
                                return '<strong>' + formatted_precision_micro + '</strong>';
                            } else {
                                return formatted_precision_micro;
                            }
                        }
                    },
                    {
                        data: 'precision_macro',
                        render: function(data, type, row) {
                            const formatted_precision_macro = row.precision_macro ? row.precision_macro.toFixed(2) : '0.00';

                            if (row.precision_macro === maxPrecision_Macro) {
                                return '<strong>' + formatted_precision_macro + '</strong>';
                            } else {
                                return formatted_precision_macro;
                            }
                        }
                    },
                    // { data: 'execute_time' },
                    {
                        data: null,
                        orderable: false,
                        render: function (data, type, row) {
                            return `
                                <div style="min-width: 100px;">
                                    <button style="margin:1px;" class="btn btn-primary" onclick="showplots(${row.id})" data-bs-toggle="tooltip" title="Show Plots"><i class="bi bi-bar-chart"></i></button>
                                    <button style="margin:1px;" class="btn btn-info" onclick="showdetails(${row.id})" data-bs-toggle="tooltip" title="Show Details"><i class="bi bi-info-circle"></i></button>
                                    <button style="margin:1px;" class="btn btn-success" onclick="showconditions(${row.id})" data-bs-toggle="tooltip" title="Show Conditions"><i class="bi bi-file-earmark-code"></i></button>
                                    <button style="margin:1px;" class="btn btn-success" onclick="showdatareport(${row.id})" data-bs-toggle="tooltip" title="Show Data Report"><i class="bi bi-file-earmark-code"></i></button>
                                </div>
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
