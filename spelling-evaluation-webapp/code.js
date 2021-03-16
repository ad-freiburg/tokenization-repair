$("document").ready(function() {
    read_results();
});

function read_results() {
    $.getJSON("results.json", function(data) {
        approaches = Object.keys(data.correct);
        fill_results_table(data);
        show_sequences(data);
    });
}

function fill_results_table(results) {
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

function show_sequences(results) {
    sequences_div = $("#sequences");
    for (var i = 0; i < results.sequences.length; i += 1) {
        sequence = results.sequences[i];
        console.log(sequence);
        table = "<p><b>Sequence " + i + "</b><table><tbody>";
        table += "<tr><td>input</td><td>" + get_labeled_token_sequence(sequence.corrupt) + "</td></tr>";
        table += "<tr><td>ground truth</td><td>" + get_labeled_token_sequence(sequence.correct) + "</td></tr>";
        for (var j = 0; j < approaches.length; j += 1) {
            table += "<tr><td>" + approaches[j] + "</td><td>" + get_labeled_token_sequence(sequence.predicted[approaches[j]]) + "</td></tr>";
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
