import { useEffect, useState } from "react";

/** Hash routing: works from file://, any sub-path and GitHub Pages with no
 *  server rewrite rules. */
export function useRoute(fallback: string): string {
  const read = () => window.location.hash.replace(/^#\/?/, "") || fallback;
  const [route, set] = useState(read);
  useEffect(() => {
    const on = () => {
      set(read());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export const go = (route: string) => {
  window.location.hash = `/${route}`;
};
