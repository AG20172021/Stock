import React, { Component } from "react";

class App extends Component{
  constructor(props){
    super(props)
    this.state = { 
      currentPage : 1,
      posts: [],
      loading: null,
      postsPerPage: 5
    };
  }
  render () {
    return (
      <div class="homePageDropDown">
        <button class="Algorithms">Algorithms</button>
      </div>
    );
  }
}
export default App;