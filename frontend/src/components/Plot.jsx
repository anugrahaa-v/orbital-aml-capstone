import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'
const P = createPlotlyComponent(Plotly)
export const layoutBase = (extra = {}) => ({
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#b8c6dd', size: 11 },
  margin: { l: 48, r: 14, t: 28, b: 40 }, legend: { orientation: 'h', y: -0.25 },
  xaxis: { gridcolor: 'rgba(120,170,255,.12)', zerolinecolor: 'rgba(120,170,255,.2)' },
  yaxis: { gridcolor: 'rgba(120,170,255,.12)', zerolinecolor: 'rgba(120,170,255,.2)' }, ...extra,
})
export default function Plot({ data, layout, height = 300, className = '' }) {
  return <P data={data} layout={layoutBase({ height, autosize: true, ...layout })} useResizeHandler style={{ width: '100%' }} className={className} config={{ displaylogo: false, responsive: true }} />
}
export const CLASS_COLORS = { PAYLOAD: '#38bdf8', DEBRIS: '#fb7185', ROCKET_BODY: '#fbbf24' }
