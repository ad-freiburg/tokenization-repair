$("document").ready(function() {
    BENCHMARK_ORDER = [
        "ACL.test.json",
        "arXiv.OCR.test.json",
        "Wiki.typos.spaces.test.json",
        "Wiki.typos.no_spaces.test.json",
    ];
    APPROACH_ORDER = [
        "google",
        "ours+post+google",
        "oracle+post+google",
        "nastase"
    ];
    get_result_files();
});

function get_result_files() {
    result_files = [];
    $.get("results/", function(data) {
        $(data).find("a").each(function() {
            name = $(this).attr("href");
            if (name.endsWith(".json")) {
                result_files.push(name);
            }
        });
        ordered_result_files = [];
        for (file_name of BENCHMARK_ORDER) {
            if (result_files.includes(file_name)) {
                ordered_result_files.push(file_name);
            }
        }
        for (file_name of result_files) {
            if (!BENCHMARK_ORDER.includes(file_name)) {
                ordered_result_files.push(file_name);
            }
        }
        result_files = ordered_result_files;
        console.log(result_files);
        for (name of result_files) {
            $("#select_results_file").append(new Option(name, name));
        }
        $("#select_results_file").val(-1);
        show_overview_table();
    });
}

function show_overview_table() {
    overview_results = {};
    n_files_read = 0;
    result_files.forEach(function(file) {
        $.get("results/" + file, function(data) {
            benchmark = file.substring(0, file.length - 5);
            overview_results[benchmark] = {};
            gt_positives = data.total.MIXED + data.total.OCR_ERROR + data.total.TOKENIZATION_ERROR;
            gt_negatives = data.total.NONE;
            keys = Object.keys(data.correct);
            for (key of keys) {
                tp = data.correct[key].MIXED + data.correct[key].OCR_ERROR + data.correct[key].TOKENIZATION_ERROR;
                fp = gt_negatives - data.correct[key].NONE;
                fn = gt_positives - tp;
                precision = tp / (tp + fp);
                recall = tp / (tp + fn);
                f1 = 2 * precision * recall / (precision + recall);
                overview_results[benchmark][key] = f1;
            }
            n_files_read += 1;
            if (n_files_read == result_files.length) {
                create_overview_table();
            }
        });
    });
}

function create_overview_table() {
    benchmarks = Object.keys(overview_results);
    approaches = new Set();
    for (benchmark of benchmarks) {
        for (approach of Object.keys(overview_results[benchmark])) {
            approaches.add(approach);
        }
    }
    approaches = Array.from(approaches);
    ordered_approaches = [];
    for (approach of APPROACH_ORDER) {
        if (approaches.includes(approach)) {
            ordered_approaches.push(approach);
        }
    }
    for (approach of approaches) {
        if (!APPROACH_ORDER.includes(approach)) {
            ordered_approaches.push(approach);
        }
    }
    approaches = ordered_approaches;
    // thead
    thead = "<tr>";
    thead += "<th>Approach</th>";
    for (benchmark of benchmarks) {
        thead += "<th>" + benchmark + "</th>";
    }
    thead += "</tr>";
    $("#thead_overview").html(thead);
    // tbody
    tbody = "";
    for (approach of approaches) {
        tbody += "<tr>";
        tbody += "<td>" + approach + "</td>";
        for (benchmark of benchmarks) {
            if (overview_results[benchmark][approach]) {
                result = overview_results[benchmark][approach];
                result = (result * 100).toFixed(2) + " %";
            } else {
                result = "-";
            }
            tbody += "<td>" + result + "</td>";
        }
        tbody += "</tr>";
    }
    $("#tbody_overview").html(tbody);
}

function read_results() {
    file = $("#select_results_file").val();
    $.getJSON("results/" + file, function(data) {
        results = data;
        approaches = Object.keys(data.correct);
        fill_results_table();
        show_sequences();
        select_sequences();
        show_error_free_sequences();
    });
}

function fill_results_table() {
    gt_tokenization = results.total.TOKENIZATION_ERROR;
    gt_ocr = results.total.OCR_ERROR;
    gt_mixed = results.total.MIXED;
    gt_total = gt_tokenization + gt_ocr + gt_mixed;
    gt_negatives = results.total.NONE;
    tbody = $("#results_table_body");
    tbody.html("");
    row = "<tr><td></td><td>ground truth</td>";
    row += "<td>" + gt_tokenization + " (" + (gt_tokenization / gt_total * 100).toFixed(1) + "%)</td>";
    row += "<td>" + gt_ocr + " (" + (gt_ocr / gt_total * 100).toFixed(1) + "%)</td>";
    row += "<td>" + gt_mixed + " (" + (gt_mixed / gt_total * 100).toFixed(1) + "%)</td>";
    row += "<td>" + (gt_ocr + gt_mixed) + " (" + ((gt_ocr + gt_mixed) / gt_total * 100).toFixed(1) + "%)</td>";
    row += "<td>" + gt_total + "</td>";
    row += "<td>-</td>";
    row += "<td>-</td>";
    row += "<td>-</td>";
    row += "<td>-</td>";
    row += "</tr>";
    tbody.append(row);
    approaches.forEach(function(approach) {
        console.log(approach);
        tokenization_correct = results.correct[approach].TOKENIZATION_ERROR;
        ocr_correct = results.correct[approach].OCR_ERROR;
        mixed_correct = results.correct[approach].MIXED;
        total_correct = tokenization_correct + ocr_correct + mixed_correct;
        false_positives = gt_negatives - results.correct[approach].NONE;
        false_negatives = gt_total - total_correct;
        precision = total_correct / (total_correct + false_positives);
        recall = total_correct / gt_total;
        f1 = 2 * precision * recall / (precision + recall);
        checkbox = create_approach_checkbox(approach);
        row = "<tr>";
        row += "<td>" + checkbox + "</td>";
        row += "<td>" + approach + "</td>";
        row += "<td>" + tokenization_correct + " (" + (tokenization_correct / gt_tokenization * 100).toFixed(1) + "%)</td>";
        row += "<td>" + ocr_correct + " (" + (ocr_correct / gt_ocr * 100).toFixed(1) + "%)</td>";
        row += "<td>" + mixed_correct + " (" + (mixed_correct / gt_mixed * 100).toFixed(1) + "%)</td>";
        row += "<td>" + (ocr_correct + mixed_correct) + " (" + ((ocr_correct + mixed_correct) / (gt_mixed + gt_ocr) * 100).toFixed(1) + "%)</td>";
        //row += "<td></td>";
        row += "<td>" + total_correct + " (" + (total_correct / gt_total * 100).toFixed(1) + "%)</td>";
        row += "<td>" + false_positives + " (" + (false_positives / gt_negatives * 100).toFixed(1) + "%)</td>";
        row += "<td>" + (precision * 100).toFixed(2) + "%</td>";
        row += "<td>" + (recall * 100).toFixed(2) + "%</td>";
        row += "<td>" + (f1 * 100).toFixed(2) + "%</td>";
        row += "</tr>";
        tbody.append(row);
    });
}

function get_checkbox_id(approach) {
    return "checkbox_" + approach;
}

function get_sequence_class(approach) {
    return "sequence_" + approach.replaceAll("+", "");
}

function create_approach_checkbox(approach) {
    checkbox_id = get_checkbox_id(approach);
    checkbox = "<input type=\"checkbox\" id=\"" + checkbox_id + "\" onchange=\"select_sequences()\">";
    return checkbox;
}

function arrays_equal(a, b) {
    if (a.length != b.length) {
        return false;
    }
    for (var i = 0; i < a.length; i += 1) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

function sequence_is_error_free(sequence_object) {
    if (!arrays_equal(sequence_object.correct.tokens, sequence_object.corrupt.tokens)) {
        return false;
    }
    for (approach of approaches) {
        if (!arrays_equal(sequence_object.correct.tokens, sequence_object.predicted[approach].tokens)) {
            return false;
        }
    }
    return true;
}

function show_sequences() {
    sequences_div = $("#sequences");
    sequences_div.html("");
    for (var i = 0; i < results.sequences.length; i += 1) {
        sequence = results.sequences[i];
        all_correct = true;
        if (sequence_is_error_free(sequence)) {
            table_class = "error_free_sequence";
        } else {
            table_class = "sequence_with_error";
        }
        table = "<p class=\"" + table_class + "\"><b>Sequence " + (i + 1) + "</b><table><tbody>";
        table += "<tr><td>input</td><td>" + get_labeled_token_sequence(sequence.corrupt, true) + "</td></tr>";
        table += "<tr><td>ground truth</td><td>" + get_labeled_token_sequence(sequence.correct, true) + "</td></tr>";
        for (var j = 0; j < approaches.length; j += 1) {
            approach = approaches[j];
            sequence_class = get_sequence_class(approach);
            predicted = sequence.predicted[approach];
            labeled_sequence = get_labeled_token_sequence(predicted);
            table += "<tr class=\"" + sequence_class + "\"><td>" + approach + "</td><td>" + labeled_sequence + "</td></tr>";
        }
        table += "</tbody></table></p>";
        sequences_div.append(table);
    }
    hide_groundtruth_labels();
}

function get_labeled_token_sequence(sequence_object, is_ground_truth = false) {
    tokens = sequence_object.tokens;
    labels = sequence_object.labels;
    html = "";
    for (var i = 0; i < tokens.length; i += 1) {
        token = tokens[i];
        label = labels[i];
        if (i > 0) {
            html += " ";
        }
        if (label == "NONE") {
            html += token;
        } else {
            classes = "tooltip " + label;
            if (is_ground_truth) {
                classes += " ground_truth";
            }
            html += "<div class=\"" + classes + "\">";
            html += token;
            html += "<span class=\"tooltiptext\">" + label + "</span>";
            html += "</div>";
        }
    };
    return html;
};

function select_sequences() {
    for (var i = 0; i < approaches.length; i += 1) {
        approach = approaches[i];
        checkbox_id = get_checkbox_id(approach);
        sequence_class_selector = "." + get_sequence_class(approach);
        if (document.getElementById(checkbox_id).checked) {
            $(sequence_class_selector).show();
        } else {
            $(sequence_class_selector).hide();
        }
    }
}

function show_error_free_sequences() {
    if (document.getElementById("checkbox_show_error_free_sequences").checked) {
        $(".error_free_sequence").show();
    } else {
        $(".error_free_sequence").hide();
    }
}

function hide_groundtruth_labels() {
    show_groundtruth_labels = document.getElementById("checkbox_show_groundtruth_labels").checked;
    if (show_groundtruth_labels) {
        elements = $(".ground_truth_hidden");
        for (el of elements) {
            el.className = el.className.replace("ground_truth_hidden", "ground_truth");
        }
    } else {
        elements = $(".ground_truth");
        for (el of elements) {
            el.className = el.className.replace("ground_truth", "ground_truth_hidden");
        }
    }
}
