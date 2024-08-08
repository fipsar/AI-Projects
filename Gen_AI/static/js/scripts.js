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


function copyContents_ques() {
  var copyText = document.querySelector(".copy-content_ques");
  var range = document.createRange();
  range.selectNode(copyText);
  window.getSelection().addRange(range);
  try {
    var successful = document.execCommand("copy");
    if (successful) {
      document.getElementById("clipboard_icon_ques").style.display = "none";
      document.getElementById("tick_icon_ques").style.display = "inline";
      setTimeout(function () {
        document.getElementById("tick_icon_ques").style.display = "none";
        document.getElementById("clipboard_icon_ques").style.display = "inline";
      }, 1500);
    }
    // var msg = successful? 'Copied!': 'Unable to copy.';
    console.log(msg);
  } catch (err) {
    console.log("Unable to copy.");
  }
  window.getSelection().removeAllRanges();
}

function openVisualization(chartFile) {
    window.open(chartFile, "_blank");
}

function enlargeContent(type) {
  console.log('Enlarge_function');
  var container = document.getElementById('vis_table_res_container');
  var content = document.getElementById(type === 'chart' ? 'visuals_side_res' : 'response_text_side');
  var isExpanded = container.style.position === 'fixed';
  var closeButton = document.getElementById('vis_close_button');

  if (isExpanded) {
      console.log('small');
      container.style.position = 'static';
      container.style.transform = 'none';
      container.style.width = '100%';
      container.style.height = '100%';
      container.style.zIndex = 'auto';
      container.style.backgroundColor = 'transparent';
      //   container.style.overflow = 'auto';
      //   container.style.top = '0px';
      //   container.style.left = '0px';
      //   container.style.right = '0px';
      //   content.style.width = '100%';
      //   content.style.height = '100%';

      if (closeButton) closeButton.remove();
  } else {
      console.log('big');
      container.style.position = 'fixed';
      container.style.padding = '0 30px  30px 30px'
      container.style.top = '0';
      container.style.left = '0';
      container.style.width = '100vw';
      container.style.height = '100vh';
      container.style.zIndex = '1000';
      container.style.backgroundColor = 'white';
      container.style.overflow = 'auto';
      content.style.width = '100%';
      content.style.height = '100%';

      if (!closeButton) {
          closeButton = document.createElement('button');
          closeButton.id = 'vis_close_button';
          closeButton.innerHTML = 'x';
          closeButton.style.position = 'absolute';
          closeButton.style.top = '5px';
          closeButton.style.right = '15px';
          closeButton.style.background = 'red';
          closeButton.style.color = 'white';
          closeButton.style.border = 'none';
          closeButton.style.padding = '5px';
          closeButton.style.cursor = 'pointer';
          closeButton.style.zIndex = '1001';
          closeButton.style.display = 'flex'
          closeButton.onclick = function () {
              enlargeContent(type);
          };
          container.appendChild(closeButton);
      }
  }
}
function downloadChart() {
  var iframe = document.getElementById('chartIframe');
  var iframeWindow = iframe.contentWindow;

  iframeWindow.document.addEventListener('DOMContentLoaded', function () {
      var chartCanvas = iframeWindow.document.querySelector('canvas');
      if (chartCanvas) {
          var url = chartCanvas.toDataURL('image/png');
          var link = document.createElement('a');
          link.href = url;
          link.download = 'chart.png';
          link.click();
      } else {
          alert('Chart not found.');
      }
  });
}

// function downloadTable() {
//   var table = document.getElementById('res_text_side');
//   var wb = XLSX.utils.table_to_book(table, { sheet: "Sheet1" });
//   var wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'binary' });

//   function s2ab(s) {
//       var buf = new ArrayBuffer(s.length);
//       var view = new Uint8Array(buf);
//       for (var i = 0; i < s.length; i++) view[i] = s.charCodeAt(i) & 0xFF;
//       return buf;
//   }

//   var blob = new Blob([s2ab(wbout)], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
//   var link = document.createElement('a');
//   link.href = URL.createObjectURL(blob);
//   link.download = 'table.xlsx';
//   link.click();
// }
function downloadTable() {
  var table = document.getElementById('res_text_side');

  var tableElement = table.querySelector('table');
  if (tableElement) {
      var wb = XLSX.utils.table_to_book(tableElement, { sheet: "Sheet1" });
      var wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'binary' });

      function s2ab(s) {
          var buf = new ArrayBuffer(s.length);
          var view = new Uint8Array(buf);
          for (var i = 0; i < s.length; i++) view[i] = s.charCodeAt(i) & 0xFF;
          return buf;
      }

      var blob = new Blob([s2ab(wbout)], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
      var link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'table.xlsx';
      link.click();
  } else {
      var textContent = table.textContent || 'No table data found.';
      var blob = new Blob([textContent], { type: 'text/plain' });
      var link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'table.txt';
      link.click();
  }
}



