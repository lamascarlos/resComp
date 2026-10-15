#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if an input file was provided
if [ -z "$1" ]; then
    echo "Error: Please provide a Markdown file."
    echo "Usage: $0 filename.md"
    exit 1
fi

INPUT_FILE="$1"

# Verify the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# Extract the base name without the .md extension
BASE_NAME="${INPUT_FILE%.*}"
OUTPUT_FILE="${BASE_NAME}.pdf"

# -------------------------------------------------------------
# CONFIGURATION / USEFUL OPTIONS
# Modify these variables to customize your document layout
# -------------------------------------------------------------
MARGINS="10mm"        # Page margins (any CSS unit)
PAPER_SIZE="A4"       # Options: A4, letter, legal...
BODY_WIDTH="100%"     # Max text width, e.g. "170mm" to narrow the column

echo "Converting '$INPUT_FILE' to '$OUTPUT_FILE'..."

# Temp files for the intermediate HTML and the GitHub-like stylesheet
TMP_DIR=$(mktemp -d /tmp/md2pdf-XXXXXX)
CSS_FILE="$TMP_DIR/style.css"
HTML_FILE="$TMP_DIR/document.html"
CHROMIUM_PROFILE="$TMP_DIR/chromium-profile"
DEVTOOLS_PORT_FILE="$CHROMIUM_PROFILE/DevToolsActivePort"
CHROMIUM_PID=""

cleanup() {
    if [ -n "$CHROMIUM_PID" ] && kill -0 "$CHROMIUM_PID" 2>/dev/null; then
        kill "$CHROMIUM_PID" 2>/dev/null || true
        wait "$CHROMIUM_PID" 2>/dev/null || true
    fi
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# GitHub-flavored markdown stylesheet so the PDF matches the rendered look
cat > "$CSS_FILE" <<CSS
@page {
    size: ${PAPER_SIZE};
    margin: ${MARGINS};
}
html {
    -webkit-print-color-adjust: exact;
}
body {
    font-family: -apple-system, "Segoe UI", "Noto Sans", Helvetica, Arial,
                 "Liberation Sans", "DejaVu Sans", sans-serif,
                 "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji";
    font-size: 9pt;
    line-height: 1.35;
    color: #1f2328;
    max-width: ${BODY_WIDTH};
    margin: 0 auto;
    word-wrap: break-word;
}
h1, h2, h3, h4, h5, h6 {
    margin-top: 12px;
    margin-bottom: 6px;
    font-weight: 600;
    line-height: 1.25;
    page-break-after: avoid;
}
h1 {
    font-size: 1.5em;
    padding-bottom: 0.3em;
    border-bottom: 1px solid #d1d9e0;
}
h2 {
    font-size: 1.15em;
    padding-bottom: 0.3em;
    border-bottom: 1px solid #d1d9e0;
}
h3 { font-size: 1.25em; }
h4 { font-size: 1em; }
h5 { font-size: 0.875em; }
h6 { font-size: 0.85em; color: #59636e; }
p, ul, ol, blockquote, table, pre {
    margin-top: 0;
    margin-bottom: 8px;
}
ul, ol { padding-left: 2em; }
li + li { margin-top: 0.25em; }
a { color: #0969da; text-decoration: none; }
strong { font-weight: 600; }
hr {
    height: 0.25em;
    padding: 0;
    margin: 24px 0;
    background-color: #d1d9e0;
    border: 0;
}
blockquote {
    padding: 0 1em;
    color: #59636e;
    border-left: 0.25em solid #d1d9e0;
    margin-left: 0;
    margin-right: 0;
    page-break-inside: avoid;
}
code, pre {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
                 "Liberation Mono", "DejaVu Sans Mono", monospace;
    font-size: 85%;
}
code {
    padding: 0.2em 0.4em;
    background-color: rgba(129, 139, 152, 0.12);
    border-radius: 6px;
}
pre {
    padding: 16px;
    overflow: auto;
    line-height: 1.45;
    background-color: #f6f8fa;
    border-radius: 6px;
    page-break-inside: avoid;
}
pre code {
    padding: 0;
    background-color: transparent;
}
table {
    border-spacing: 0;
    border-collapse: collapse;
    width: max-content;
    max-width: 100%;
}
th, td {
    padding: 3px 8px;
    border: 1px solid #d1d9e0;
}
th { font-weight: 600; }
tr:nth-child(2n) { background-color: #f6f8fa; }
img { max-width: 100%; }
CSS

# Step 1: Markdown -> standalone HTML (GitHub-flavored Markdown input)
pandoc "$INPUT_FILE" \
    -f gfm+tex_math_dollars \
    -t html5 \
    --standalone \
    --embed-resources \
    --mathml \
    --css="$CSS_FILE" \
    --metadata title="$(basename "$BASE_NAME")" \
    -V include-before='' \
    -o "$HTML_FILE"

# Pandoc's --metadata title injects a visible <h1 class="title">; remove it
# so the PDF starts with the document's own heading.
sed -i 's|<h1 class="title">[^<]*</h1>||' "$HTML_FILE"

# Step 2: HTML -> PDF through Chromium DevTools, enabling a custom footer.
chromium-browser \
    --headless \
    --disable-gpu \
    --no-first-run \
    --no-default-browser-check \
    --remote-allow-origins='*' \
    --remote-debugging-port=0 \
    --user-data-dir="$CHROMIUM_PROFILE" \
    "$HTML_FILE" > "$TMP_DIR/chromium.log" 2>&1 &
CHROMIUM_PID=$!

for _ in $(seq 1 100); do
    if [ -s "$DEVTOOLS_PORT_FILE" ]; then
        break
    fi
    if ! kill -0 "$CHROMIUM_PID" 2>/dev/null; then
        cat "$TMP_DIR/chromium.log" >&2
        exit 1
    fi
    sleep 0.1
done

if [ ! -s "$DEVTOOLS_PORT_FILE" ]; then
    echo "Error: Timed out waiting for Chromium DevTools." >&2
    exit 1
fi

DEVTOOLS_PORT=$(head -n 1 "$DEVTOOLS_PORT_FILE")

node - "$DEVTOOLS_PORT" "$HTML_FILE" "$OUTPUT_FILE" <<'NODE'
const fs = require('fs');
const http = require('http');
const { pathToFileURL } = require('url');

const [port, inputPath, outputPath] = process.argv.slice(2);

function getJson(path) {
    return new Promise((resolve, reject) => {
        http.get({ host: '127.0.0.1', port, path }, (response) => {
            let body = '';
            response.setEncoding('utf8');
            response.on('data', (chunk) => { body += chunk; });
            response.on('end', () => {
                if (response.statusCode !== 200) {
                    reject(new Error(`DevTools ${path} returned ${response.statusCode}`));
                    return;
                }
                resolve(JSON.parse(body));
            });
        }).on('error', reject);
    });
}

async function main() {
    const targets = await getJson('/json/list');
    const target = targets.find(({ type }) => type === 'page');
    if (!target) throw new Error('Chromium did not create a page target');

    const socket = new WebSocket(target.webSocketDebuggerUrl);
    const pending = new Map();
    let messageId = 0;

    const send = (method, params = {}) => new Promise((resolve, reject) => {
        const id = ++messageId;
        pending.set(id, { resolve, reject });
        socket.send(JSON.stringify({ id, method, params }));
    });

    socket.addEventListener('message', ({ data }) => {
        const message = JSON.parse(data);
        if (!message.id) return;
        const request = pending.get(message.id);
        if (!request) return;
        pending.delete(message.id);
        if (message.error) request.reject(new Error(message.error.message));
        else request.resolve(message.result);
    });

    await new Promise((resolve, reject) => {
        socket.addEventListener('open', resolve, { once: true });
        socket.addEventListener('error', reject, { once: true });
    });

    await send('Page.enable');
    await send('Page.navigate', { url: pathToFileURL(inputPath).href });
    for (let attempt = 0; attempt < 100; attempt += 1) {
        const { result } = await send('Runtime.evaluate', { expression: 'document.readyState' });
        if (result.value === 'complete') break;
        await new Promise((resolve) => setTimeout(resolve, 100));
    }

    const { data } = await send('Page.printToPDF', {
        displayHeaderFooter: true,
        headerTemplate: '<div></div>',
        footerTemplate: '<div style="width:100%;text-align:center;font-size:8px"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
        preferCSSPageSize: true,
        printBackground: true,
    });
    fs.writeFileSync(outputPath, Buffer.from(data, 'base64'));
    socket.close();
}

main().catch((error) => {
    console.error(error.stack || error.message);
    process.exitCode = 1;
});
NODE

echo "Success! Created '$OUTPUT_FILE'."
