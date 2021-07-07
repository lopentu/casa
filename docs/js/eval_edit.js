
Vue.component('eval-edit', {
  props: ['evaldata'],
  computed: {          
      pos: function() { return (this.evaldata && this.evaldata.pos) || "" },
      neg: function() { return (this.evaldata && this.evaldata.neg) || "" } 
  },
  methods: {
    update_eval: function(category, value) { 
      new_data = Object.assign({}, {pos: this.pos, neg: this.neg});
      new_data[category] = value;
      this.$emit('evalupdate', new_data);
    }
  },  
  template: ` 
  <div class="eval-edit">
    <div class="eval-edit-item eval-edit-pos">
      <textarea class="ta-pos" v-bind:value="pos" 
        v-on:input="update_eval('pos', $event.target.value)" 
        placeholder="補充正面詞彙"></textarea>
    </div>
    <div class="eval-edit-item eval-edit-neg">
      <textarea class="ta-neg" v-bind:value="neg" 
      v-on:input="update_eval('neg', $event.target.value)"
      placeholder="補充負面詞彙"></textarea>
    </div>
  </div>
  `
})

var app = new Vue({
    el: '#app',
    created: function(){        
        fetch('data/eval_ontology_raw_210707.json')            
            .then(resp=>resp.text())
            .then(text=>{
                text = text.replace(/NaN/g, "null");
                data = JSON.parse(text);                
                this.onto = Object.keys(data)
                  .sort((x,y)=>data[y]["evals"].length-data[x]["evals"].length)
                  .reduce((obj, key)=>{
                    obj[key] = data[key];
                    return obj;
                  }, {});
            });
        localforage.getItem('editor')
            .then((value)=> this.pageId=value || "first-editor" );
        localforage.getItem('augdata')
            .then((value)=>{
              this.augdata = value || {};
              this.status = "已載入"
            }).catch((err)=>{              
              console.error(err);
            });
    },
    
    data: {    
      pageId: "first-editor",
      status: "載入中",
      onto: {},      
      augdata: {"[通訊]國內電信漫遊": {pos: "positive", neg: "negative "}},
    },

    methods: {
      update_augdata: function(category, newdata){        
        this.augdata[category] = newdata;        
        localforage.setItem('augdata', this.augdata)
          .then((val)=>{
            let pad = (x)=> x.toString().padStart(2, '0');
            let t = new Date();
            let timestamp = 
              `${t.getFullYear()}/${pad(t.getMonth()+1)}/${pad(t.getDate())} 
              ${t.getHours()}:${pad(t.getMinutes()+1)}:${pad(t.getSeconds())}`;
            this.status="上次儲存: " + timestamp;
          })
          .catch((err)=>{
            console.error(err);
          });
      },

      onPageIdUpdate: function() {
        localforage.setItem('editor', this.pageId)
          .catch((err)=> console.error(err));
      },

      export_data: function(){
        let blob = new Blob([JSON.stringify(this.augdata, null, 2)], 
                  {type: 'text/plain;charset=utf-8'});
        let t = new Date();
        let pad = (x)=> x.toString().padStart(2, '0');
        let timestamp = `${t.getFullYear()}${pad(t.getMonth()+1)}${pad(t.getDate())}`;
        saveAs(blob, `${this.pageId}_aug_data_${timestamp}.json`);

      }
    }
  })
