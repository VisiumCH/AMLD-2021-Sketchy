import React from "react";
import { Switch, Route } from "react-router-dom";
import Dataset from "./interactionPages/seeDataset.js";
import Drawing from "./interactionPages/drawing.js";
import Embeddings from "./interactionPages/embeddingsPlot.js";
import ScalarPerformance from "./performancePages/scalarPerformance.js";
import ImagePerformance from "./performancePages/imagePerformance.js";

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
