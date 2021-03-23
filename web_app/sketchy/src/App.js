import React from "react";
import { Switch, Route } from "react-router-dom";
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
    </Switch>
  );
};

export default App;
