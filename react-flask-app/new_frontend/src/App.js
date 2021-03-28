import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import {
  BrowserRouter as Router,
  Switch,
  Route,
} from "react-router-dom";

// Solid test articles
/**
 * CNN: https://www.cnn.com/2021/02/05/media/lou-dobbs-fox-show-canceled/index.html
 * FOX: https://www.foxnews.com/politics/biden-terrorist-designation-yemens-houthi-militia
 * HUFFPOST: https://www.huffpost.com/entry/covid-19-eviction-crisis-women_n_5fca8af3c5b626e08a29de11
 */

//Components:
import Homepage from './Homepage/Homepage';
import Deets from './Deets/Deets';
import AboutUs from './AboutUs/AboutUs';

function App() {
  return (
    <Router>
      <Switch>
        <Route exact path="/">
          <div className="App">
            <Homepage />
          </div>
        </Route>
        <Route exact path="/about-us">
          <div className="App">
            <AboutUs />
          </div>
        </Route>
        <Route exact path="/deets">
          <div className="App">
            <Deets />
          </div>
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
