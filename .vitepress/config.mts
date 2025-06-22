import { defineConfig } from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "nlp",
  description: "自然语言处理专题网站",
  srcDir: "src",
  lang: "zh-CN",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "主页", link: "/" },
      { text: "文档", link: "/总览" },
    ],
    docFooter: {
      prev: "上一页",
      next: "下一页",
    },
    returnToTopLabel: "回到顶部",
    sidebarMenuLabel: "菜单",
    darkModeSwitchLabel: "主题",
    lightModeSwitchTitle: "切换到浅色模式",
    darkModeSwitchTitle: "切换到深色模式",

    sidebar: [
      {
        text: "nlp models",
        items: [
          { text: "总览", link: "/总览" },
          { text: "Word2Vec", link: "/Word2Vec" },
          { text: "Seq2Seq", link: "/Seq2Seq" },
          { text: "Transformer", link: "/Transformer" },
          { text: "GPT", link: "/GPT" },
          // { text: "Word2Vec", link: "/Word2Vec" },
          // { text: "Seq2Seq", link: "/Seq2Seq" },
        ],
      },
    ],

    socialLinks: [
      { icon: "github", link: "https://github.com/vuejs/vitepress" },
    ],
  },
  vite: {
    server: {
      host: "127.0.0.1",
      port: 5011,
    },
    preview: {
      host: "127.0.0.1",
      port: 5010,
    },
  },
});
