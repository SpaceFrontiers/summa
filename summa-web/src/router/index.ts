import { createRouter, createWebHashHistory } from "vue-router";

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "search",
      component: () => import("../views/SearchView.vue"),
      props: (route) => ({ query: route.query }),
    },
    {
      path: "/databases",
      name: "databases",
      component: () => import("../views/DatabasesView.vue"),
    },
    {
      path: "/about",
      name: "about",
      component: () => import("../views/AboutView.vue"),
    },
  ],
});

export default router;
