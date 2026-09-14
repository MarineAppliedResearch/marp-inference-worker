#ifndef PayloadDir
  #error PayloadDir is required
#endif
#ifndef OutputDir
  #error OutputDir is required
#endif
#ifndef WorkerVersion
  #error WorkerVersion is required
#endif
#ifndef CoordinatorUrl
  #error CoordinatorUrl is required
#endif
#ifndef EnrollmentCode
  #error EnrollmentCode is required
#endif

[Setup]
AppId={{C46A4328-55A7-46B1-A277-5FC4DC63BC91}
AppName=MARP Inference Worker
AppVersion={#WorkerVersion}
DefaultDirName={localappdata}\MARP\Worker
DefaultGroupName=MARP Inference Worker
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=marp-inference-worker-{#WorkerVersion}-windows-x64-setup
Compression=lzma2/ultra64
SolidCompression=yes
Uninstallable=yes

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{tmp}\marp-worker-payload"; Flags: deleteafterinstall recursesubdirs createallsubdirs

[Icons]
Name: "{group}\MARP Worker"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File &quot;{app}\launcher.ps1&quot; -Screen window"
Name: "{group}\MARP Worker (fullscreen)"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File &quot;{app}\launcher.ps1&quot; -Screen fullscreen"
Name: "{group}\Finish current MARP work, then pause"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File &quot;{app}\launcher.ps1&quot; -Control finish"
Name: "{group}\Stop MARP work now, then pause"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File &quot;{app}\launcher.ps1&quot; -Control stop"
Name: "{group}\Resume MARP work"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File &quot;{app}\launcher.ps1&quot; -Control resume"
Name: "{userstartup}\MARP Inference Worker"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File &quot;{app}\launcher.ps1&quot; -Screen window"

[Run]
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File &quot;{app}\launcher.ps1&quot; -Screen window"; Description: "Start the MARP worker"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
function GetCoordinatorUrl(Param: String): String;
begin
  Result := '{#CoordinatorUrl}';
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  SaveStringToFile(ExpandConstant('{tmp}\marp-worker-activation.txt'), '{#EnrollmentCode}', False);
  Result := '';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Parameters: String;
begin
  if CurStep <> ssPostInstall then
    Exit;

  Parameters := ExpandConstant(
    '-NoProfile -ExecutionPolicy Bypass -File "{tmp}\marp-worker-payload\bootstrap-windows.ps1" ' +
    '-PayloadRoot "{tmp}\marp-worker-payload" -InstallRoot "{app}" ' +
    '-CoordinatorUrl "{code:GetCoordinatorUrl}" ' +
    '-ActivationCodeFile "{tmp}\marp-worker-activation.txt"');
  WizardForm.StatusLabel.Caption := 'Downloading and preparing the MARP worker. This can take several minutes...';
  SaveStringToFile(ExpandConstant('{app}\setup.log'),
    'Starting MARP worker bootstrap.' + #13#10, False);

  if not Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Parameters,
      '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
    RaiseException('Windows could not start the MARP worker setup process.');

  if ResultCode <> 0 then
  begin
    SaveStringToFile(ExpandConstant('{app}\setup.log'),
      'Bootstrap process exited with code ' + IntToStr(ResultCode) + '.' + #13#10, True);
    MsgBox('MARP worker setup failed. Error details are saved in:' + #13#10 +
      ExpandConstant('{app}\setup.log'), mbError, MB_OK);
    RaiseException('MARP worker setup failed.');
  end;
end;
