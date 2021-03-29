$("document").ready(function() {
    read_results();
});

function read_results() {
    $.getJSON("results.json", function(data) {
        results = data;
        approaches = Object.keys(data.correct);
        fill_results_table();
        create_approach_checkboxes();
        show_sequences();
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
    row = "<tr><td>ground truth</td>";
    row += "<td>" + gt_tokenization + "</td>";
    row += "<td>" + gt_ocr + "</td>";
    row += "<td>" + gt_mixed + "</td>";
    row += "<td>" + gt_total + "</td>";
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
        row = "<tr><td>" + approach + "</td>";
        row += "<td>" + tokenization_correct + " (" + (tokenization_correct / gt_tokenization * 100).toFixed(1) + "%)</td>";
        row += "<td>" + ocr_correct + " (" + (ocr_correct / gt_ocr * 100).toFixed(1) + "%)</td>";
        row += "<td>" + mixed_correct + " (" + (mixed_correct / gt_mixed * 100).toFixed(1) + "%)</td>";
        row += "<td>" + total_correct + " (" + (total_correct / gt_total * 100).toFixed(1) + "%)</td>";
        row += "<td>" + false_positives + " (" + (false_positives / gt_negatives * 100).toFixed(1) + "%)</td>";
        tbody.append(row);
    });
}

function get_checkbox_id(approach) {
    return "checkbox_" + approach;
}

function get_sequence_class(approach) {
    return "sequence_" + approach.replaceAll("+", "");
}

function create_approach_checkboxes() {
    selectors_div = $("#approach_checkboxes");
    for (var i = 0; i < approaches.length; i += 1) {
        approach = approaches[i];
        checkbox_id = get_checkbox_id(approach);
        checkbox = "<input type=\"checkbox\" id=\"" + checkbox_id + "\" checked onchange=\"select_sequences()\">";
        html = checkbox + " " + approach + "<br>";
        selectors_div.append(html);
    }
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
        table += "<tr><td>input</td><td>" + get_labeled_token_sequence(sequence.corrupt) + "</td></tr>";
        table += "<tr><td>ground truth</td><td>" + get_labeled_token_sequence(sequence.correct) + "</td></tr>";
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
}

function get_labeled_token_sequence(sequence_object) {
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
            html += "<div class=\"tooltip " + label + "\">";
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
