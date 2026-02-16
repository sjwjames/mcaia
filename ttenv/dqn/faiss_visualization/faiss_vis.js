import { Feder } from "@zilliz/feder";


const filePath =
  "https://assets.zilliz.com/faiss_ivf_flat_voc_17k_ab112eec72.index";
const source = "faiss"; // or hnswlib
const feder = new Feder({
  filePath,
  source,
});


const overviewDom = feder.overview();
const searchViewDom = feder.search(vector);