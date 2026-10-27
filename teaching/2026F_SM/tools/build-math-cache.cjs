// Build reusable inline SVG formulas from the course page's authoritative LaTeX.
// Requires Node.js, Playwright, and a Chromium browser; no source formulas are changed.
// Result is embedded in teaching_2026_SM.html. Re-run after editing formulas.
// Verified build/slow-network results are recorded in the adjacent README.md.
const fs=require('fs'),path=require('path'),http=require('http'),crypto=require('crypto');
const {chromium}=require(process.env.SM_PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'../../..'),file=path.join(root,'teaching_2026_SM.html');
const begin='<!-- SM-PRECOMPILED-MATH:BEGIN -->',end='<!-- SM-PRECOMPILED-MATH:END -->';
const original=fs.readFileSync(file,'utf8');
const uncache=s=>s.replace(/<!-- SM-PRECOMPILED-MATH:BEGIN -->[\s\S]*?<!-- SM-PRECOMPILED-MATH:END -->\n?/,'');
const source=uncache(original).replace(/<script id="sm-mathjax-cache-start">[\s\S]*?<\/script>/,'<script>window.smRenderMath();</script>');
const server=http.createServer((req,res)=>{
 const p=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
 if(!p.startsWith(root+'/'))return res.writeHead(403).end();
 if(p===file){res.setHeader('Content-Type','text/html; charset=utf-8');return res.end(source);}
 fs.readFile(p,(e,data)=>{if(e)return res.writeHead(404).end();res.setHeader('Content-Type',({'.js':'text/javascript','.html':'text/html','.css':'text/css','.svg':'image/svg+xml'})[path.extname(p)]||'application/octet-stream');res.end(data)});
});
(async()=>{let browser;try{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 browser=await chromium.launch({headless:true,...(process.env.SM_BROWSER?{executablePath:process.env.SM_BROWSER}:{})});
 const page=await browser.newPage();
 await page.route('**/*',route=>new URL(route.request().url()).hostname==='127.0.0.1'?route.continue():route.abort());
 await page.goto('http://127.0.0.1:'+server.address().port+'/teaching_2026_SM.html',{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>!!window.MathJax?.tex2svgPromise,{timeout:60000});
 await page.evaluate(async()=>{await MathJax.startup.promise;await window.smRenderMath();});
 const rendered=await page.evaluate(()=>({
  css:document.getElementById('MJX-SVG-styles')?.textContent,
  formulas:[...document.querySelectorAll('mjx-container.sm-rendered-math')].map(el=>({tex:el.dataset.tex,display:el.getAttribute('display')==='true',html:el.outerHTML})),
  errors:document.querySelectorAll('mjx-merror').length,
  pending:document.querySelectorAll('math[data-tex],span.sm-inline-math[data-tex]').length
 }));
 if(rendered.errors||rendered.pending||!rendered.css||!rendered.formulas.length)throw Error(JSON.stringify({errors:rendered.errors,pending:rendered.pending,count:rendered.formulas.length}));
 const glyphs=new Map(),formulas={};
 for(const f of rendered.formulas){
  let markup=f.html.replace(/^<mjx-container\b[^>]*>/,tag=>tag.replace(/\s(?:id|data-tex|data-sm-precompiled)="[^"]*"/g,''));
  const local=new Map();
  markup=markup.replace(/<defs>([\s\S]*?)<\/defs>/g,(_,defs)=>{
   const paths=[...defs.matchAll(/<path\b[^>]*\bid="([^"]+)"[^>]*><\/path>/g)];
   if(paths.map(m=>m[0]).join('')!==defs)throw Error('Unsupported SVG defs structure; preserve glyph definitions explicitly.');
   for(const m of paths){const shape=m[0].replace(/\sid="[^"]*"/,'');const id='sm-glyph-'+crypto.createHash('sha256').update(shape).digest('hex').slice(0,20);local.set(m[1],id);glyphs.set(id,shape.replace('<path','<path id="'+id+'"'));}
   return '';
  });
  markup=markup.replace(/((?:xlink:)?href)="#([^"]+)"/g,(_,attr,id)=>attr+'="#'+(local.get(id)||id)+'"');
  formulas[JSON.stringify([f.tex,f.display])]=markup;
 }
 const payload=JSON.stringify(formulas).replace(/</g,'\\u003c');
 const block=begin+'\n<style id="sm-cached-math-styles">'+rendered.css+'</style>\n'+
  '<svg id="sm-cached-math-glyphs" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" style="position:absolute;width:0;height:0;overflow:hidden"><defs>'+[...glyphs.values()].join('')+'</defs></svg>\n'+
  '<script type="application/json" id="sm-math-cache">'+payload+'</script>\n'+end+'\n';
 if(fs.readFileSync(file,'utf8')!==original)throw Error('Source changed during build; retry without overwriting edits.');
 const clean=uncache(original);fs.writeFileSync(file,clean.replace('<script id="sm-mathjax-config">',block+'<script id="sm-mathjax-config">'));
 console.log(JSON.stringify({occurrences:rendered.formulas.length,unique:Object.keys(formulas).length,glyphs:glyphs.size,cacheBytes:Buffer.byteLength(block)}));
 }finally{await browser?.close();await new Promise(r=>server.close(r));}})().catch(e=>{console.error(e);process.exitCode=1});
