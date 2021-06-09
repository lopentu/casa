

var app = new Vue({
    el: '#app',
    created: function(){
        console.log("heelo");
        fetch('data/eval_ontology_raw_210605.json')
            .then(resp=>resp.json())
            .then(data=>{
                console.log(data);
                this.onto = data;
            });
    },
    
    data: {
      message: 'Hello Vue!',
      onto: {},      
      show_ngrams: true,
      show_words: false,
      show_evals: false
    },

    methods: {

    }
  })