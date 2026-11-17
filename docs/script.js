
// 1. Display the content from content.js

var boxes = document.querySelectorAll('[data-content]');

for (var i = 0; i < boxes.length; i++) {
    var box = boxes[i];
    var name = box.dataset.content;
    var text = CONTENT[name];

    if (text) {
        box.innerHTML = text;
    }
}


// 2. Display the figures

var figures = document.querySelectorAll('[data-figures]');

for (var i = 0; i < figures.length; i++) {
    var box = figures[i];
    var items = box.dataset.figures.split(';');

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
