export const classes = ['No leak', 'LOCA', 'LOCAC', 'SGATR', 'SGBTR', 'SLBIC', 'FLB', 'LLB'] as const;
export const places: Record<string, string> = {'No leak':'No leak indicated', LOCA:'Hot-leg pipe', LOCAC:'Cold-leg pipe', SGATR:'Steam generator A tubes', SGBTR:'Steam generator B tubes', SLBIC:'Steam line', FLB:'Feedwater line', LLB:'Letdown line'};
export type Prediction = {time:number;score:number;hits:number;alarm:boolean;component:string;location_component:string;locations:Record<string,number>;disagreement:boolean};
export type Event = {time:number;kind:string;message:string;component:string};
export type Replay = {name:string;illustrative:boolean;model_id:string|null;model_name:string;threshold:number;persistence:number;sample_interval:number;time_steps:number;stride:number;location_classes:string[];location_conditional:boolean;localization?:{policy:string;clip_times:number[];votes:string[];tied:boolean;available_at:number}|null;sensors:{key:string;label:string;unit:string}[];readings:Record<string,number>[];predictions:Prediction[];events:Event[];row_count:number;start:number;end:number};
export type Metrics = {recordings:number;detected_leak_runs:number;missed_leak_runs:number;false_alarm_runs:number;quiet_nonleak_runs:number;location_accuracy:number|null;median_delay_seconds:number|null;delay_detected_runs:number;location_evaluation_scope?:string};
export type Evaluation = {baseline:Metrics;candidate:Metrics;scope:string;test_groups:string[]};
export type Model = {id:string;name:string;kind:string;created:string;features:string[];classes:string[];time_steps:number;stride:number;sample_interval:number;threshold:number;persistence:number;parent_id?:string;evaluation?:Evaluation};
export type Registry = {active:string|null;versions:Model[]};
export type Recording = {id:string;name:string;rows:number;start:number;end:number};
export type TrainingRow = Recording & {group:string;split:string;label:string;onset:string;origin:string};
export type Job = {id:string;status:string;phase:string;progress:number;error?:string;model_id?:string;evaluation?:Evaluation;history?:{epoch:number;validation_loss:number}[]};

export async function api<T>(path:string, init?:RequestInit):Promise<T> {
  let response:Response;
  try { response = await fetch(`/api${path}`, init); }
  catch { throw new Error('Cannot reach the local model service. Start the backend and try again.'); }
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The request could not be processed. Check the selected files and settings.');
  return result as T;
}
export function elapsed(time:number) { const s=Math.max(0,Math.floor(time)); return `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`; }
export function illustrativeReplay():Replay {
  const readings:Record<string,number>[]=[]; const predictions:Prediction[]=[]; const events:Event[]=[];
  const sensors=[{key:'P',label:'Primary pressure',unit:'bar'},{key:'TAVG',label:'Coolant temperature',unit:'°C'},{key:'WRCA',label:'Loop A flow',unit:'t/h'},{key:'RM1',label:'Radiation monitor',unit:'CPM'}];
  for(let time=0; time<=900; time+=10) {
    const leak=Math.max(0,Math.min(1,(time-300)/220));
    readings.push({time,P:155-12*leak+Math.sin(time/42)*.7,TAVG:305+4*leak+Math.sin(time/55)*1.2,WRCA:21000-1800*leak+Math.sin(time/28)*160,RM1:40+120*leak+Math.sin(time/33)*3});
  }
  let hits=0,alarm=false;
  for(let time=110;time<=900;time+=50) {
    const score=Math.min(.96,.07+Math.max(0,time-300)/350);
    hits=score>=.7?hits+1:0;
    const component=score>.4?'SGATR':'No leak'; const loc:Record<string,number>={};
    classes.forEach(c=>{loc[c]=c===component?.79:.03;});
    if(hits===1) events.push({time,kind:'elevated',message:'Illustrative score crossed threshold',component});
    if(hits>=3&&!alarm) {alarm=true;events.push({time,kind:'alarm',message:'Illustrative alarm confirmed',component});}
    predictions.push({time,score,hits,alarm,component,location_component:component,locations:loc,disagreement:(score>=.7)!==(component!=='No leak')});
  }
  return {name:'Steam generator A · illustrative scenario',illustrative:true,model_id:null,model_name:'Illustrative values · no model connected',threshold:.7,persistence:3,sample_interval:10,time_steps:12,stride:5,location_classes:[...classes],location_conditional:false,sensors,readings,predictions,events,row_count:readings.length,start:0,end:900};
}
