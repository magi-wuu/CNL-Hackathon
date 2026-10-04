import {Component, type ReactNode, type ComponentRef, Suspense, useMemo, useRef, useEffect} from 'react';
import {Canvas, type ThreeEvent} from '@react-three/fiber';
import {OrbitControls, Grid} from '@react-three/drei';
import {CanvasTexture, CatmullRomCurve3, Vector3, DoubleSide} from 'three';
import {RotateCcw, Move, Box} from 'lucide-react';
import {places} from './types';

type Props={alarmPart:string|null; selected:string; onSelect:(id:string)=>void};
const steel='#718b99',gold='#c5ae73',water='#24748a';

function Label({text,position}:{text:string;position:[number,number,number]}) {
  const texture=useMemo(()=>{
    const canvas=document.createElement('canvas');canvas.width=512;canvas.height=96;
    const context=canvas.getContext('2d')!;
    context.fillStyle='#a9c3cf';context.font='500 32px monospace';context.textAlign='center';context.fillText(text,256,52);
    context.strokeStyle='#466271';context.beginPath();context.moveTo(64,68);context.lineTo(448,68);context.stroke();
    return new CanvasTexture(canvas);
  },[text]);
  useEffect(()=>()=>texture.dispose(),[texture]);
  return <sprite position={position} scale={[2.15,.4,1]} renderOrder={9}><spriteMaterial map={texture} transparent depthTest={false} depthWrite={false}/></sprite>;
}

function Pipe({points,id,color=gold,...props}:Props & {points:number[][];id:string;color?:string}) {
  const curve=useMemo(()=>new CatmullRomCurve3(points.map(p=>new Vector3(p[0],p[1],p[2]))),[points]);
  const hot=props.alarmPart===id,selected=props.selected===id;
  return <mesh onClick={(e:ThreeEvent<MouseEvent>)=>{e.stopPropagation();props.onSelect(id);}} onPointerOver={()=>{document.body.style.cursor='pointer';}} onPointerOut={()=>{document.body.style.cursor='auto';}}>
    <tubeGeometry args={[curve,40,.095,10,false]}/><meshStandardMaterial color={hot?'#ff5367':selected?'#8bdef0':color} metalness={.5} roughness={.32} emissive={hot?'#d12b3e':selected?'#165a6b':'#000'} emissiveIntensity={hot?.65:.2}/>
  </mesh>;
}
function Vessel({x=0,y=0,r=.78,h=5.2,id,props}:{x?:number;y?:number;r?:number;h?:number;id:string;props:Props}) {
  const hot=props.alarmPart===id,selected=props.selected===id;
  const color=hot?'#ff5367':selected?'#92c6d6':steel;
  return <group position={[x,y,0]} onClick={e=>{e.stopPropagation();props.onSelect(id);}}>
    <mesh><cylinderGeometry args={[r,r,h,48,1,true,1.5,Math.PI*1.32]}/><meshStandardMaterial color={color} metalness={.7} roughness={.33} side={DoubleSide} emissive={hot?'#d12b3e':'#000'} emissiveIntensity={.5}/></mesh>
    {[-h/2,h/2].map(yy=><mesh key={yy} position={[0,yy,0]} scale={[1,.38,1]}><sphereGeometry args={[r,40,20]}/><meshStandardMaterial color={color} metalness={.65} roughness={.34}/></mesh>)}
    {[-h*.42,-h*.17,h*.17,h*.42].map(yy=><mesh key={yy} position={[0,yy,0]} rotation={[Math.PI/2,0,0]}><torusGeometry args={[r+.005,.045,10,48]}/><meshStandardMaterial color='#a0adb0' metalness={.8} roughness={.25}/></mesh>)}
    <mesh><cylinderGeometry args={[r*.75,r*.75,h*.83,32,1,true,1.5,Math.PI*1.32]}/><meshStandardMaterial color={water} transparent opacity={.32} side={DoubleSide} metalness={.2}/></mesh>
  </group>;
}
function Geometry(props:Props) {
  const rods=Array.from({length:13},(_,i)=>[Math.sin(i*2.4)*.29,Math.cos(i*2.4)*.29]);
  return <group position={[0,.25,0]}>
    <Vessel id='vessel' props={props}/>
    <group onClick={e=>{e.stopPropagation();props.onSelect('core');}}>
      {rods.map(([x,z],i)=><mesh key={i} position={[x,-1.4,z]}><cylinderGeometry args={[.045,.045,1.45,10]}/><meshStandardMaterial color={gold} metalness={.45} roughness={.35}/></mesh>)}
    </group>
    <group onClick={e=>{e.stopPropagation();props.onSelect('rods');}}>
      {rods.slice(0,7).map(([x,z],i)=><mesh key={i} position={[x,1,z]}><cylinderGeometry args={[.026,.026,4.2,8]}/><meshStandardMaterial color='#bba88a' metalness={.7} roughness={.25}/></mesh>)}
      <mesh position={[0,3.05,0]}><cylinderGeometry args={[.52,.52,.16,32]}/><meshStandardMaterial color={steel} metalness={.7}/></mesh>
    </group>
    {[-2.25,2.25].map((x,i)=><group key={x}>
      <Label position={[x,2.95,0]} text={`GENERATOR ${i===0?'A':'B'}`}/>
      <Vessel x={x} y={.35} r={.48} h={3.35} id={i===0?'SGATR':'SGBTR'} props={props}/>
      <group onClick={e=>{e.stopPropagation();props.onSelect(i===0?'SGATR':'SGBTR');}}>
        {[-.18,0,.18].map(z=><Pipe key={z} points={[[x-.13,1.45,z],[x-.13,-.75,z],[x,-1,z],[x+.13,-.75,z],[x+.13,1.45,z]]} id={i===0?'SGATR':'SGBTR'} {...props} color={gold}/>)}
      </group>
      <Pipe points={[[Math.sign(x)*.6,.3,0],[x*.55,.5,0],[x,.5,0]]} id='LOCA' {...props} color='#d08b62'/>
      <Pipe points={[[x,-.65,0],[x*.65,-1.5,.35],[Math.sign(x)*.6,-1.6,.1]]} id='LOCAC' {...props} color='#469bb7'/>
      <mesh position={[x*.65,-1.4,.35]} rotation={[Math.PI/2,0,0]} onClick={e=>{e.stopPropagation();props.onSelect('pumps');}}><cylinderGeometry args={[.23,.23,.45,24]}/><meshStandardMaterial color={steel} metalness={.7}/></mesh>
      <Pipe points={[[x,2.15,0],[x,2.65,0],[x+Math.sign(x)*.55,2.7,0]]} id='SLBIC' {...props}/>
      <Pipe points={[[x+Math.sign(x)*.9,-1.7,-.15],[x+Math.sign(x)*.5,-1.3,-.15],[x,-.8,-.15]]} id='FLB' {...props} color='#62a8ac'/>
    </group>)}
    <Vessel x={1.12} y={2.1} r={.23} h={1.1} id='pressurizer' props={props}/>
    <Pipe points={[[.62,.8,-.25],[1.1,1.25,-.25],[1.12,1.8,0]]} id='pressurizer' {...props}/>
    <Pipe points={[[-.65,-1.8,.2],[-1.15,-2.05,.45],[-1.3,-2.65,.6],[-2.1,-2.65,.6]]} id='LLB' {...props} color='#7b8fd4'/>
    <mesh position={[0,-2.87,0]}><cylinderGeometry args={[.93,1.05,.22,40]}/><meshStandardMaterial color='#3a505f' metalness={.5}/></mesh>
    {[-.48,.48].map(x=><mesh key={x} position={[x,3.35,-.42]}><cylinderGeometry args={[.025,.025,.7,8]}/><meshStandardMaterial color={steel}/></mesh>)}
    <Pipe points={[[-.48,3.65,-.42],[0,3.65,-.42],[.48,3.65,-.42]]} id='rods' {...props} color={steel}/>
  </group>;
}
class CanvasBoundary extends Component<{children:ReactNode},{failed:boolean}> {
  state={failed:false};
  static getDerivedStateFromError(){return {failed:true};}
  render(){return this.state.failed?<div className='canvas-fallback'><Box/><p>3D view unavailable</p><small>Enable WebGL in your browser. Component findings remain available below.</small></div>:this.props.children;}
}
export default function Reactor(props:Props) {
  const controls=useRef<ComponentRef<typeof OrbitControls>>(null);
  const extra:Record<string,string>={vessel:'Reactor vessel',core:'Reactor core',rods:'Control rods',pumps:'Coolant pumps',pressurizer:'Pressurizer'};
  return <div className='reactor-view'>
    <div className='reactor-watermark'>PWR<span>SCHEMATIC / CUTAWAY</span></div>
    <CanvasBoundary><Canvas camera={{position:[7,4.2,12],fov:37}} dpr={[1,1.7]} fallback={<div className='canvas-fallback'>This browser does not support WebGL.</div>}>
      <color attach='background' args={['#0e171e']}/><ambientLight intensity={1.2}/><directionalLight position={[4,8,8]} intensity={3.2} color='#ccedff'/><directionalLight position={[-5,2,-3]} intensity={2} color='#53c6db'/>
      <Suspense fallback={null}><Geometry {...props}/><Grid position={[0,-2.8,0]} args={[14,14]} cellSize={.5} cellThickness={.4} cellColor='#203b4b' sectionSize={2} sectionThickness={.6} sectionColor='#375362' fadeDistance={18} infiniteGrid/><OrbitControls ref={controls} makeDefault minDistance={6} maxDistance={19} target={[0,.3,0]} maxPolarAngle={Math.PI*.75} enableDamping/></Suspense>
    </Canvas></CanvasBoundary>
    <div className='canvas-tools'><span><Move size={13}/> Drag to orbit · scroll to zoom</span><button aria-label='Reset reactor camera' onClick={()=>controls.current?.reset()}><RotateCcw size={15}/></button></div>
    <div className='component-details'><div><span className='eyebrow'>SELECTED COMPONENT</span><strong>{places[props.selected]||extra[props.selected]||'Reactor vessel'}</strong></div><span className={`tag ${props.alarmPart===props.selected?'danger':'muted'}`}>{props.alarmPart===props.selected?'Suspected issue':'Explore'}</span></div>
  </div>;
}
