import { createRouter, createWebHashHistory } from "vue-router";

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "search",
      component: () => import("../views/SearchView.vue"),
      props: route => ({ query: route.query })
    }, {
      path: "/manage",
      name: "manage",
      component: () => import("../views/ManageView.vue"),
    },
  ],
});

export default router;
