const target = new URL("./index.html", window.location.href);
const current = new URL(window.location.href);
target.searchParams.set("view", "mobile");
target.searchParams.set("level", "pursuitArcade");
for (const name of ["boss", "round"]) {
  if (current.searchParams.has(name)) target.searchParams.set(name, current.searchParams.get(name));
}
window.location.replace(target);
