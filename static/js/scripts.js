function copyContentsAndMore() {
  copyContents();
  copycontentss();
}
function copyContent(button) {
  var copyText = document.querySelector(".copy-content");
  var range = document.createRange();
  range.selectNode(copyText);
  window.getSelection().addRange(range);
  try {
    var successful = document.execCommand("copy");
    if (successful) {
      document.getElementById("clipboard_icon_sql").style.display = "none";
      document.getElementById("tick_icon_sql").style.display = "inline";
      setTimeout(function () {
        document.getElementById("tick_icon_sql").style.display = "none";
        document.getElementById("clipboard_icon_sql").style.display = "inline";
      }, 1500);
    }

    // var msg = successful ? 'Copied!': 'Unable to copy.';
    console.log(msg);
    if (button) {
      var icon = querySelector("i");
      if (icon) {
        icon.classList.remove("fa-clone");
        icon.classList.add("fa-check");
      }
    }
  } catch (err) {
    console.log("Unable to copy.");
  }
  window.getSelection().removeAllRanges();
}

function copyContents() {
  var copyText = document.querySelector(".copy-contents");
  var range = document.createRange();
  range.selectNode(copyText);
  window.getSelection().addRange(range);
  try {
    var successful = document.execCommand("copy");
    if (successful) {
      document.getElementById("clipboard_icon").style.display = "none";
      document.getElementById("tick_icon").style.display = "inline";
      setTimeout(function () {
        document.getElementById("tick_icon").style.display = "none";
        document.getElementById("clipboard_icon").style.display = "inline";
      }, 1500);
    }
    // var msg = successful? 'Copied!': 'Unable to copy.';
    console.log(msg);
  } catch (err) {
    console.log("Unable to copy.");
  }
  window.getSelection().removeAllRanges();
}

function copycontentss() {
  var copyText = document.querySelector(".dataframe");
  var range = document.createRange();
  range.selectNode(copyText);
  window.getSelection().addRange(range);
  try {
    var successful = document.execCommand("copy");
    if (successful) {
      document.getElementById("clipboard_icon").style.display = "none";
      document.getElementById("tick_icon").style.display = "inline";
      setTimeout(function () {
        document.getElementById("tick_icon").style.display = "none";
        document.getElementById("clipboard_icon").style.display = "inline";
      }, 1500);
    }
    // var msg = successful ? 'Copied!': 'Unable to copy.';
    console.log(msg);
  } catch (err) {
    console.log("Unable to copy.");
  }
  window.getSelection().removeAllRanges();
}

function openVisualization(chartFile) {
    window.open(chartFile, "_blank");
}


