// Inlines css/style.css + js/core.js + js/widgets.js into index.html between
// <!--CSS--> / <!--JS--> markers, producing a fully self-contained page.
// Run after editing any source file:  node build.mjs
import{readFileSync as rd,writeFileSync as wr}from'fs';
let html=rd('index.html','utf8');
const css=rd('css/style.css','utf8');
const core=rd('js/core.js','utf8').replace(/^export /gm,'');
const data=rd('js/realdata.js','utf8');
const widg=rd('js/widgets.js','utf8').replace(/^import.*$/m,'// (core.js + realdata.js inlined above)');
html=html.replace(/<!--CSS-->[\s\S]*?<!--\/CSS-->/,`<!--CSS--><style>\n${css}</style><!--/CSS-->`);
html=html.replace(/<!--JS-->[\s\S]*?<!--\/JS-->/,`<!--JS--><script type="module">\n${core}\n${data}\n${widg}</script><!--/JS-->`);
wr('index.html',html);
console.log('built: index.html is self-contained ('+(html.length/1024).toFixed(0)+' KB)');
