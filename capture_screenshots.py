import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1440, 'height': 900})
        page = await context.new_page()
        
        # --- 1. MLflow Screenshots ---
        print('Capturing MLflow...')
        # Experiment List
        await page.goto('http://localhost:5000/#/experiments/658536374581253365')
        await page.wait_for_timeout(3000)
        await page.screenshot(path='Membangun_model/screenshoot_experiment.jpg')
        await page.screenshot(path='Membangun_model/screenshoot_dashboard.jpg') # Duplicate as dashboard
        
        # Artifacts
        # Click the first run
        await page.goto('http://localhost:5000/#/experiments/658536374581253365/runs/ce50faa272464799a362961d37db7ffc/artifacts')
        await page.wait_for_timeout(3000)
        await page.screenshot(path='Membangun_model/screenshoot_artifacts_tab.jpg')
        await page.screenshot(path='Membangun_model/screenshoot_artifak.jpg') 
        
        # --- 2. Serving Proof ---
        print('Capturing Serving Proof...')
        await page.goto('http://localhost:8000/metrics')
        await page.wait_for_timeout(1000)
        await page.screenshot(path='Monitoring dan Logging/1.bukti_serving.jpg')
        
        # --- 3. Prometheus Screenshots ---
        print('Capturing Prometheus...')
        await page.goto('http://localhost:9090/targets')
        await page.wait_for_timeout(2000)
        await page.screenshot(path='Monitoring dan Logging/4.bukti monitoring Prometheus/1.monitoring_targets.jpg')
        
        await page.goto('http://localhost:9090/graph?g0.expr=model_accuracy_current&g0.tab=0')
        await page.wait_for_timeout(2000)
        await page.screenshot(path='Monitoring dan Logging/4.bukti monitoring Prometheus/2.monitoring_accuracy.jpg')
        
        await page.goto('http://localhost:9090/graph?g0.expr=http_request_duration_seconds_sum&g0.tab=0')
        await page.wait_for_timeout(2000)
        await page.screenshot(path='Monitoring dan Logging/4.bukti monitoring Prometheus/3.monitoring_latency.jpg')
        
        await page.goto('http://localhost:9090/graph?g0.expr=model_inference_errors_total&g0.tab=0')
        await page.wait_for_timeout(2000)
        await page.screenshot(path='Monitoring dan Logging/4.bukti monitoring Prometheus/4.monitoring_errors.jpg')
        
        await page.goto('http://localhost:9090/graph?g0.expr=system_cpu_usage&g0.tab=0')
        await page.wait_for_timeout(2000)
        await page.screenshot(path='Monitoring dan Logging/4.bukti monitoring Prometheus/5.monitoring_cpu.jpg')
        
        # --- 4. Grafana Screenshots ---
        print('Capturing Grafana...')
        await page.goto('http://localhost:3000/login')
        try:
            await page.fill('input[name="user"]', 'dicoding')
            await page.fill('input[name="password"]', 'admin')
            await page.click('button[type="submit"]')
            await page.wait_for_timeout(3000)
        except Exception as e:
            print("Login step skipped or failed:", e)
            pass
            
        await page.goto('http://localhost:3000/')
        await page.wait_for_timeout(2000)
        await page.screenshot(path='Monitoring dan Logging/5.bukti monitoring Grafana/1.monitoring_home.jpg')
        
        await page.goto('http://localhost:3000/explore?left=%7B%22datasource%22:%22Prometheus%22,%22queries%22:%5B%7B%22expr%22:%22http_requests_total%22,%22refId%22:%22A%22%7D%5D,%22range%22:%7B%22from%22:%22now-1h%22,%22to%22:%22now%22%7D%7D')
        await page.wait_for_timeout(3000)
        await page.screenshot(path='Monitoring dan Logging/5.bukti monitoring Grafana/2.monitoring_explore.jpg')
        
        await page.goto('http://localhost:3000/d/heart_disease_azzar/heart-disease-monitoring-azzar-budiyanto')
        await page.wait_for_timeout(4000)
        await page.screenshot(path='Monitoring dan Logging/5.bukti monitoring Grafana/3.monitoring_new_dashboard.jpg')
        
        await page.goto('http://localhost:3000/alerting/list')
        await page.wait_for_timeout(3000)
        await page.screenshot(path='Monitoring dan Logging/6.bukti alerting Grafana/1.alerting_list.jpg')
        
        await browser.close()
        print('All screenshots successfully generated and overwritten!')

asyncio.run(run())
