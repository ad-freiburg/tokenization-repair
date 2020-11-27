$(document).ready(function() {
    console.log("document ready");
    
    // GET BENCHMARKS
    
    benchmarks = [];
    
    $.get("../benchmarks/", function(data) {
        $(data).find("a").each(function() {
            name = $(this).attr("href");
            if (!name.endsWith(".txt")) {
                name = name.substring(0, name.length - 1);
                benchmarks.push(name);
		console.log(name);
                $("#select_benchmark").append(new Option(name, name));
            }
        });
        set_prediction_options()
    });
    
    // GET PREDICTION FILES
    
    $("#select_benchmark").change(function() {
        set_prediction_options();
    })
    
    $("#select_subset").change(function() {
        set_prediction_options();
    })
    
    $("#select_predictions").change(function() {
        create_table();
    })
    
    $("#ignore_punctuation").change(function() {
        create_table();
    })
    
    $("#hide_zeros").change(function () {
        hide_zero_rows();
    })
});

function set_prediction_options() {
    benchmark = $("#select_benchmark option:selected").val();
    console.log(benchmark);
    
    subset = $("#select_subset option:selected").val();
    console.log(subset);
    
    prediction_files = [];
    $("#select_predictions").empty();
    
    read_benchmark();
    
    results_dir = "../results/" + benchmark + "/" + subset + "/";
    $.get(results_dir, function(data) {
        $(data).find("a").each(function() {
            name = $(this).attr("href");
            prediction_files.push(name);
            $("#select_predictions").append(new Option(name, name));
        });
        $("#select_predictions").prop("selectedIndex", -1);
        $("#table").html("select a file above");
    });
}

function read_benchmark() {
    benchmark_dir = "../benchmarks/" + benchmark + "/" + subset + "/";
    $.get(benchmark_dir + "correct.txt", function(data) {
        correct_sequences = []
        for (sequence of data.split("\n")) {
            if (sequence.length > 0) {
                correct_sequences.push(sequence);
            }
        }
    });
    $.get(benchmark_dir + "corrupt.txt", function(data) {
        corrupt_sequences = data.split("\n");
    });
}

function create_table() {
    $("#table").html("evaluating...");
    
    selected = $("#select_predictions option:selected").val();
    predictions_file = results_dir + selected;
    console.log(predictions_file);
    
    $.get(predictions_file, function(data) {
        predicted_sequences = data.split("\n");
        
        n = Math.min(correct_sequences.length, corrupt_sequences.length, predicted_sequences.length);
        console.log(n);
        
        n_corrupt = 0;
        n_correct = 0;
        
        n_sequences = 0;
        n_tp = 0;
        n_fp = 0;
        n_fn = 0;
        
        table = "<table>\n";
        table += "<th>ID</th>"
        table += "<th>INPUT</th>"
        table += "<th>GROUND TRUTH</th>"
        table += "<th>PREDICTED</th>"
        table += "<th>TP</th>"
        table += "<th>FP</th>"
        table += "<th>FN</th>"
        table += "<th>CORR</th>\n"
        
        for (var i = 0; i < n; i++) {
            if (corrupt_sequences[i].replaceAll(' ', '') == predicted_sequences[i].replaceAll(' ', '')) {
                n_sequences += 1;
                
                // ground truth and prediction
                
                [ground_truth, highlight_true] = get_differences(corrupt_sequences[i], correct_sequences[i]);
                correct_highlighted = highlight_positions(correct_sequences[i], highlight_true);
                [predictions, highlight_predicted] = get_differences(corrupt_sequences[i], predicted_sequences[i]);
                [_unused, wrong_positions] = get_differences(correct_sequences[i], predicted_sequences[i]);
                predicted_highlighted = highlight_positions_with_truth(predicted_sequences[i], highlight_predicted, wrong_positions);
                
                // evaluate TP, FP, FN
                
                tp = ground_truth.filter(x => predictions.includes(x));
                fp = predictions.filter(x => !ground_truth.includes(x));
                fn = ground_truth.filter(x => !predictions.includes(x));
                
                is_corrupted = tp.length + fn.length > 0;
                is_correct = fp.length + fn.length == 0;
                
                n_tp += tp.length;
                n_fp += fp.length;
                n_fn += fn.length;
                
                if (is_corrupted) {
                    n_corrupt += 1;
                }
                if (is_correct) {
                    n_correct += 1;
                }
                
                // row class
                
                row_class = null;
                if (tp.length == 0 && fp.length == 0 && fn.length == 0) {
                    row_class = "all_zeros";
                }
                
                // colors
                
                corrupt_color = "black";
                predicted_color = "black";
                if (is_corrupted) {
                    corrupt_color = "red";
                    if (is_correct) {
                        predicted_color = "green";
                    }
                }
                if (!is_correct) {
                    predicted_color = "red";
                }
                
                // sequence result
                
                if (is_correct) {
                    sequence_result = "yes";
                } else {
                    sequence_result = "no";
                }
                
                // row
                if (row_class == null) {
                    row = "<tr>";
                } else {
                    row = "<tr class=\"" + row_class + "\">";
                }
                
                // .. sequences
                row += "<td>" + i + "</td>";
                row += "<td style=\"color:" + corrupt_color + "\">" + corrupt_sequences[i] + "</td>";
                row += "<td>" + correct_highlighted + "</td>";
                row += "<td>" + predicted_highlighted + "</td>";
                // .. evaluation counts
                row += "<td>" + tp.length + "</td>";
                row += "<td>" + fp.length + "</td>";
                row += "<td>" + fn.length + "</td>";
                // .. sequence result
                row += "<td style=\"color:" + predicted_color + "\">" + sequence_result + "</td>";
                // .. end row
                row += "</tr>";
                table += row + "\n";
            }
        }
        table += "</table>";
        
        precision = n_tp / (n_tp + n_fp);
        recall = n_tp / (n_tp + n_fn);
        f1 = 2 * precision * recall / (precision + recall);
        
        evaluation = "corrupt sequences: " + (n_corrupt / n_sequences).toFixed(4) + " (" + n_corrupt + "/" + n_sequences + ")<br>\n";
        evaluation += "sequence accuracy: " + (n_correct / n_sequences).toFixed(4) + " (" + n_correct + "/" + n_sequences + ")<br>\n";
        evaluation += "true positives: &nbsp;" + n_tp + "<br>\n";
        evaluation += "false positives: " + n_fp + "<br>\n";
        evaluation += "false negatives: " + n_fn + "<br>\n";
        evaluation += "precision: " + precision.toFixed(4) + "<br>\n";
        evaluation += "recall: &nbsp;&nbsp;&nbsp;" + recall.toFixed(4) + "<br>\n";
        evaluation += "F1-score: &nbsp;" + f1.toFixed(4) + "<br>\n";
        evaluation += "<br>\n" + table;
        
        $("#table").html(evaluation);
        hide_zero_rows();
    });
}

function isalnum(char) {
    return char.match(/^[0-9a-z]+$/);
}

function get_differences(a, b) {
    ignore_punctuation = $("#ignore_punctuation").is(":checked");
    diff_positions_a = [];
    diff_positions_b = [];
    var i = 0;
    var j = 0;
    while (i < a.length && j < b.length) {
        if (a[i] == b[j]) {
            i += 1;
            j += 1;
        } else {
            do_ignore = false;
            if (ignore_punctuation) {
                if (a[i] == " ") {
                    if (!isalnum(a[i-1]) || !isalnum(a[i+1])) {
                        do_ignore = true;
                    }
                } else {
                    if (!isalnum(b[j-1]) || !isalnum(b[j+1])) {
                        do_ignore = true;
                    }
                }
            }
            if (!do_ignore) {
                diff_positions_a.push(i);
                diff_positions_b.push(j);
            }
            if (a[i] == " ") {
                i += 1;
            } else {
                j += 1;
            } 
        }
    }
    return [diff_positions_a, diff_positions_b];
}

function highlight_positions(text, positions) {
    for (pos of positions.reverse()) {
        text = text.substring(0, pos) + "<u>" + text[pos] + "</u>" + text.substring(pos + 1);
    }
    return text;
}

function union(set1, set2) {
    _union = new Set(set1);
    for (elem of set2) {
        _union.add(elem);
    }
    return _union;
}

function compare_numbers(a, b) {
    return a - b;
}

function highlight_positions_with_truth(text, predicted_positions, wrong_positions) {
    predicted_positions = new Set(predicted_positions);
    wrong_positions = new Set(wrong_positions);
    all_positions = union(predicted_positions, wrong_positions);
    all_positions = Array.from(all_positions).sort(compare_numbers);
    if (benchmark == "nastase") {
        console.log(predicted_positions);
        console.log(wrong_positions);
        console.log(all_positions);
    }
    for (pos of all_positions.reverse()) {
        if (wrong_positions.has(pos)) {
            color = "red";
        } else {
            color = "green";
        }
        if (predicted_positions.has(pos)) {
            html_element = "u"
        } else {
            html_element = "span"
        }
        highlighted = "<" + html_element + " style=\"background-color:" + color + "\">" + text[pos] + "</" + html_element + ">"
        text = text.substring(0, pos) + highlighted + text.substring(pos + 1);
    }
    return text;
}

function hide_zero_rows() {
    if ($("#hide_zeros").is(":checked")) {
        $(".all_zeros").hide();
    } else {
        $(".all_zeros").show();
    }
}
