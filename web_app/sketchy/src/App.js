import React from 'react'
import { Switch, Route } from 'react-router-dom';
import Drawing from './drawing.js'
import Embeddings from './embeddings.js'


const App = () => {

  return (
    <Switch>
      <Route exact path="/">
        <Drawing />
      </Route>
      <Route path="/embeddings">
        <Embeddings />
      </Route>
    </Switch>
  )
}


export default App
