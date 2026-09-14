#ifndef PayloadDir
  #error PayloadDir is required
#endif
#ifndef OutputDir
  #error OutputDir is required
#endif
#ifndef WorkerVersion
  #error WorkerVersion is required
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
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File &quot;{tmp}\marp-worker-payload\bootstrap-windows.ps1&quot; -PayloadRoot &quot;{tmp}\marp-worker-payload&quot; -InstallRoot &quot;{app}&quot; -CoordinatorUrl &quot;{code:GetCoordinatorUrl}&quot; -ActivationCodeFile &quot;{tmp}\marp-worker-activation.txt&quot;"; StatusMsg: "Downloading and preparing the MARP worker. This can take several minutes..."; Flags: waituntilterminated
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File &quot;{app}\launcher.ps1&quot; -Screen window"; Description: "Start the MARP worker"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
var
  ConnectPage: TInputQueryWizardPage;

procedure InitializeWizard;
begin
  ConnectPage := CreateInputQueryPage(wpSelectDir,
    'Connect this worker to MARP',
    'Enter the MARP API address and the one-time activation code.',
    'Setup downloads and installs every required component automatically.');
  ConnectPage.Add('MARP API URL:', False);
  ConnectPage.Add('Activation code:', True);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = ConnectPage.ID then
  begin
    if (Trim(ConnectPage.Values[0]) = '') or (Trim(ConnectPage.Values[1]) = '') then
    begin
      MsgBox('Both the MARP API URL and activation code are required.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

function GetCoordinatorUrl(Param: String): String;
begin
  Result := Trim(ConnectPage.Values[0]);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  SaveStringToFile(ExpandConstant('{tmp}\marp-worker-activation.txt'), Trim(ConnectPage.Values[1]), False);
  Result := '';
end;
