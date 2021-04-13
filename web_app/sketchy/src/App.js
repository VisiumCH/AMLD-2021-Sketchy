import React from "react";
import { Switch, Route, Redirect } from "react-router-dom";
import Dataset from "./seeDataset.js";
import Drawing from "./drawing.js";
import Embeddings from "./embeddingsPlot.js";

const App = () => {
  return (
    <Switch>
      <Route path="/drawing">
        <Drawing />
      </Route>
      <Route path="/embeddings">
        <Embeddings />
      </Route>
      <Route path="/">
        <Dataset />
      </Route>
    </Switch>
  );
};

export default App;
