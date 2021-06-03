import React from "react";
import { Switch, Route } from "react-router-dom";
import Dataset from "./seeDataset.js";
import Drawing from "./drawing.js";
import Embeddings from "./embeddingsPlot.js";
import ScalarPerformance from "./scalarPerformance.js";
import ImagePerformance from "./imagePerformance.js";

const App = () => {
  return (
    <Switch>
      <Route path="/drawing">
        <Drawing />
      </Route>
      <Route path="/embeddings">
        <Embeddings />
      </Route>
      <Route path="/scalar_perf">
        <ScalarPerformance />
      </Route>
      <Route path="/image_perf">
        <ImagePerformance />
      </Route>
      <Route path="/">
        <Dataset />
      </Route>
    </Switch>
  );
};

export default App;
