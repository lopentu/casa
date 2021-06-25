

var app = new Vue({
    el: '#app',
    created: function(){
        console.log("heelo");
        fetch('data/eval_ontology_raw_210605.json')            
            .then(resp=>resp.text())
            .then(text=>{
                text = text.replace(/NaN/g, "null");

                data = JSON.parse(text);
                console.log(data);
                this.onto = data;
            });
    },
    
    data: {      
      onto: {},      
      show_ngrams: true,
      show_words: false,
      show_evals: true
    },

    methods: {

    }
  })