import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import {
  BrowserRouter as Router,
  Switch,
  Route,
} from "react-router-dom";


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
