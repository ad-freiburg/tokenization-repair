$("document").ready(function() {
    console.log("ready");
    raw_dir = "corpus/raw/";
    repaired_dir = "corpus/raw.repaired_hyphens/";
    file_select = document.getElementById("file");
    files = [];
    get_files();
});

function get_files() {
    console.log("get files");
    $.get(repaired_dir, function(folder_data) {
        $(folder_data).find("a").each(function() {
            file_name = $(this).attr("href");
            if (file_name.endsWith(".txt")) {
                files.push(file_name);
            }
        });
    }).then(function() {
        for (file of files) {
            var option = document.createElement("option");
            option.text = file;
            option.value = file;
            file_select.add(option);
        }
        show_file();
    });
}

function show_file() {
    file = file_select.value;
    console.log(file);
    $.get(raw_dir + file, function(raw_data) {
        raw_lines = raw_data.split("\n");
    }).then(function() {
        $.get(repaired_dir + file, function(repaired_data) {
            repaired_lines = repaired_data.split("\n");
        }).then(function() {
            raw_html = "";
            repaired_html = "";
            for (i in raw_lines) {
                is_changed = raw_lines[i].trim() != repaired_lines[i];
                if (is_changed) {
                    raw_html += "<div style=\"color:red\">";
                    repaired_html += "<div style=\"color:red\">";
                    console.log(raw_lines[i].trim(), repaired_lines[i]);
                }
                raw_html += raw_lines[i].replace("  ", " &nbsp;") + "<br>";
                repaired_html += repaired_lines[i] + "<br>";
                if (is_changed) {
                    raw_html += "</div>";
                    repaired_html += "</div>";
                }
            }
            $("#textfield_left").html(raw_html);
            $("#textfield_right").html(repaired_html);
        });
    });
}
