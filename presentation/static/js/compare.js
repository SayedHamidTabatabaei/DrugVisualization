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

function fill_comparing_plot(comparing_plot_type, radio_group){

    const plot_name = comparing_plot_type;

    if(radio_group){
        comparing_plot_type = document.querySelector(`input[name=${radio_group}]:checked`).value;
    }

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
            const plot = document.getElementById(`${plot_name}-plot`);

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
