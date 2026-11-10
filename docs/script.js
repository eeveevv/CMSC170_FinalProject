// 1. display the content from content.js
var boxes = document.querySelectorAll('[data-content]');

for (var i = 0; i < boxes.length; i++) {
    var box = boxes[i];
    var name = box.dataset.content;
    var text = CONTENT[name];
    if (text) {
        box.innerHTML = text;
    }
}

// 2. load the result tables
var tables = document.querySelectorAll('[data-include]');
for (var i = 0; i < tables.length; i++) {
    var box = tables[i];

    fetch(box.dataset.include)
        .then(function(response) {
            if (!response.ok) {
                throw new Error(response.status);
            }
            return response.text();
        })
        .then(function(html) {
            box.innerHTML = html;
        })
        .catch(function() {
            box.textContent = 'Not available yet. Run Comparison.py first.';
        });
}

// 3. display the figures
var figures = document.querySelectorAll('[data-figures]');

for (var i = 0; i < figures.length; i++) {
    var box = figures[i];
    var items = box.dataset.figures.split(',');

    for (var j = 0; j < items.length; j++) {
        var parts = items[j].split('|');

        var fig = document.createElement('figure');
        var img = document.createElement('img');

        img.src = parts[0];
        img.alt = parts[1];

        img.onerror = function() {
            this.parentElement.remove();
        };
        var caption = document.createElement('figcaption');
        caption.textContent = parts[1];

        fig.appendChild(img);
        fig.appendChild(caption);

        box.appendChild(fig);
    }
}